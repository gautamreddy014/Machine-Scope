"""
Optional end-to-end UI test (needs: pip install playwright && playwright install chromium).
Start the server first (python backend/app.py), then run:  python backend/tests/ui_test_playwright.py
"""
import asyncio, re, os, tempfile
from playwright.async_api import async_playwright, expect
BASE=os.environ.get("MACHINESCOPE_URL","http://127.0.0.1:5000/")
D=tempfile.mkdtemp(prefix="ms_ui_")+os.sep
OUT=D
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
results=[]
def ok(name, cond, extra=""):
    results.append((name, bool(cond), extra)); print(("PASS " if cond else "FAIL ")+name, extra)

def make_fixtures():
    with open(os.path.join(ROOT,"data","ai4i2020.csv"),encoding="utf-8-sig") as f:
        lines=f.read().splitlines()
    open(D+"subset_ai4i.csv","w").write("\n".join(lines[:3001]))
    open(D+"bad.csv","w").write("a,b,c\n1,2,3\n")

async def main():
    make_fixtures()
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width":1440,"height":900}, accept_downloads=True)
        pg = await ctx.new_page()
        errs=[]
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
        await pg.goto(BASE+"index.html"); await pg.wait_for_timeout(3500)
        ok("KPI total", (await pg.text_content("#kTotal"))=="10,000")
        acc=await pg.text_content("#kAcc"); ok("KPI accuracy is % from API", re.match(r"\d+\.\d\d%", acc), acc)
        ok("confusion matrix rendered", await pg.locator("#cmGrid .cell").count()==4)
        ok("ROC chart rendered", await pg.locator("#chROC .main-svg").count()>0)
        ok("flagged table rows", await pg.locator("#flaggedTable tbody tr").count()==8)
        # prediction
        await pg.click("[data-example=strain]"); await pg.click("#predictBtn"); await pg.wait_for_selector("#predictResult .verdict")
        v=await pg.text_content("#predictResult .verdict h3"); ok("prediction high strain", "predicted" in v, v)
        await pg.click("[data-example=normal]"); await pg.click("#predictBtn"); await pg.wait_for_timeout(800)
        v2=await pg.text_content("#predictResult .verdict h3"); ok("prediction typical", v2=="No failure predicted", v2)
        await pg.fill("#fTorque",""); await pg.fill("#fSpeed","9999"); await pg.click("#predictBtn"); await pg.wait_for_timeout(300)
        e1=await pg.locator("#predictForm .err-msg").all_text_contents(); ok("client validation", "Required" in e1 and any("between" in x for x in e1), str([x for x in e1 if x]))
        # server-side validation via API directly
        r=await pg.evaluate("""fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({Type:'Z',Air_Temperature:'x'})}).then(r=>r.status)""")
        ok("server validation 422", r==422)
        await pg.click("#resetPredict"); await pg.wait_for_timeout(200)
        ok("reset clears inputs", await pg.input_value("#fAir")=="")
        await pg.screenshot(path=OUT+"t_after_predict.png")
        # anomalies
        await pg.select_option("#anFeature","Torque"); await pg.wait_for_timeout(700)
        t=await pg.text_content("#anStats"); ok("anomaly torque", "69" in t, t[:60])
        await pg.select_option("#anFeature","all"); await pg.wait_for_timeout(700)
        ok("anomaly all table", await pg.locator("#anTable tbody tr").count()>10)
        async with pg.expect_download() as dl: await pg.click("#anExport")
        d=await dl.value; path=D+"dl_anom.csv"; await d.save_as(path)
        lines=open(path).read().splitlines(); ok("anomaly CSV export", len(lines)>400 and "note" in lines[0], f"{len(lines)-1} rows")
        # maintenance
        c0=await pg.text_content("#mtCount")
        await pg.select_option("#mtPriority","High"); await pg.wait_for_timeout(500)
        c1=await pg.text_content("#mtCount"); ok("priority filter", "342" in c1, c1)
        await pg.fill("#mtSearch","L47"); await pg.wait_for_timeout(900)
        c2=await pg.text_content("#mtCount"); ok("search filter", c2!=c1, c2)
        await pg.fill("#mtSearch",""); await pg.wait_for_timeout(900)
        await pg.locator("#mtTable .review-box").first.check(); await pg.wait_for_timeout(200)
        ok("mark reviewed", "1 reviewed" in await pg.text_content("#mtReviewedCount"))
        await pg.click("label:has(#mtHideReviewed) span"); await pg.wait_for_timeout(600)
        c3=await pg.text_content("#mtCount"); ok("hide reviewed", "341" in c3, c3)
        await pg.click("#mtTable th[data-sort=Torque]"); await pg.wait_for_timeout(600)
        tq=[float(x) for x in await pg.locator("#mtTable tbody tr td:nth-child(8)").all_text_contents()]
        ok("sort by torque desc", tq==sorted(tq, reverse=True) and len(tq)>5)
        await pg.click("#mtNext"); await pg.wait_for_timeout(500)
        ok("pagination", (await pg.text_content("#mtPage")).startswith("2 /"))
        async with pg.expect_download() as dl: await pg.click("#mtExport")
        d=await dl.value; path=D+"dl_mt.csv"; await d.save_as(path)
        lines=open(path).read().splitlines(); ok("maintenance CSV export", len(lines)==342 and "reviewed_this_session" in lines[0], f"{len(lines)-1} rows")
        # run analysis
        await pg.click("#runBtn"); await pg.wait_for_selector(".toast.success:has-text('Analysis complete')", timeout=60000)
        ok("run analysis toast", True)
        # upload bad
        await pg.set_input_files("#uploadInput", D+"bad.csv"); await pg.wait_for_selector(".toast.error", timeout=20000)
        ok("bad CSV rejected", "Missing columns" in await pg.locator(".toast.error").last.text_content())
        await pg.set_input_files("#uploadInput", D+"subset_ai4i.csv"); await pg.wait_for_function("document.querySelector('#kTotal').textContent==='3,000'", timeout=120000)
        ok("valid CSV upload retrains", True, await pg.text_content("#kAcc"))
        await pg.click("#resetDataBtn"); await pg.wait_for_function("document.querySelector('#kTotal').textContent==='10,000'", timeout=60000)
        ok("reset to bundled dataset", True)
        # nav
        await pg.click(".main-nav a[href='analysis.html']"); await pg.wait_for_load_state(); await pg.wait_for_timeout(3500)
        ok("navigate to analysis", pg.url.endswith("analysis.html"))
        st=await pg.text_content("#uStats"); ok("univariate stats", "Mean" in st or "MEAN" in st.upper())
        await pg.select_option("#gType","H"); await pg.wait_for_timeout(1500)
        ok("global filter count", "1,003" in await pg.text_content("#gCount"), await pg.text_content("#gCount"))
        await pg.select_option("#gType","all"); await pg.wait_for_timeout(1200)
        await pg.select_option("#uFeature","Type"); await pg.wait_for_timeout(600)
        ok("categorical univariate", await pg.locator("#uCategorical").is_visible())
        await pg.select_option("#uFeature","Tool_Wear"); await pg.wait_for_timeout(600)
        await pg.click("#uHistMode button[data-mode=label]"); await pg.wait_for_timeout(400)
        ok("hist split mode", await pg.locator("#chUHist .trace.bars").count()>=2)
        await pg.select_option("#uBins","80"); await pg.wait_for_timeout(500)
        await pg.click(".tab[data-target=bivariate]"); await pg.wait_for_timeout(900)
        ok("tab nav bivariate active", "active" in (await pg.get_attribute(".tab[data-target=bivariate]","class")))
        for pre in ["Rotational_Speed,Torque","Machine_Failure,Tool_Wear","Type,Machine_Failure","Air_Temperature,Process_Temperature"]:
            await pg.click(f"[data-preset='{pre}']"); await pg.wait_for_timeout(800)
            ok(f"preset {pre}", await pg.locator("#chBMain .main-svg").count()>0, await pg.text_content("#bMainSub"))
        await pg.click("#bColor button[data-color=type]"); await pg.wait_for_timeout(400)
        await pg.select_option("#bGroupFeature","Power"); await pg.wait_for_timeout(700)
        ok("grouped box", "Mechanical power" in await pg.text_content("#bGroupSub"))
        await pg.click(".tab[data-target=multivariate]"); await pg.wait_for_timeout(900)
        await pg.click("#mMethod button[data-method=spearman]"); await pg.wait_for_timeout(1200)
        ok("spearman", "Spearman" in await pg.text_content("#mHeatSub"))
        await pg.click("#mFeatures label:has-text('Mechanical power')"); await pg.wait_for_timeout(1500)
        ok("feature toggle", "of 6" in await pg.text_content("#mSplomSub"), await pg.text_content("#mSplomSub"))
        await pg.select_option("#mC","Power"); await pg.wait_for_timeout(400)
        async with pg.expect_download() as dl: await pg.click("#exportFiltered")
        d=await dl.value; path=D+"dl_rec.csv"; await d.save_as(path)
        ok("filtered records CSV", len(open(path).read().splitlines())==10001)
        await pg.screenshot(path=OUT+"t_analysis_end.png", full_page=False)
        real=[e for e in errs if "422" not in e and "400 (BAD" not in e and "WebGL" not in e and "GL Driver" not in e and "swiftshader" not in e.lower()]
        ok("no JS errors", not real, str(real[:5]))
        # mobile
        m = await b.new_context(viewport={"width":390,"height":844}, device_scale_factor=1)
        mp = await m.new_page()
        for page in ["index.html","analysis.html"]:
            await mp.goto(BASE+page); await mp.wait_for_timeout(3500)
            w=await mp.evaluate("document.documentElement.scrollWidth"); ok(f"mobile no h-scroll {page}", w<=390, str(w))
            await mp.screenshot(path=OUT+f"m_{page}.png", full_page=True)
        await mp.click("#menuBtn"); await mp.wait_for_timeout(300)
        ok("mobile menu opens", await mp.locator(".main-nav").is_visible())
        await mp.screenshot(path=OUT+"m_menu.png")
        await b.close()
    print(f"\n{sum(r[1] for r in results)}/{len(results)} passed")
asyncio.run(main())
