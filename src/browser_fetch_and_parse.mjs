import process from "node:process";

import { JSDOM } from "jsdom";
import { chromium } from "playwright";
import { Defuddle } from "defuddle/node";

const inputUrl = process.argv[2];

if (!inputUrl) {
  console.error("Missing URL argument");
  process.exit(1);
}

let browser;

try {
  browser = await chromium.launch({
    channel: "chrome",
    headless: true,
  });

  const context = await browser.newContext({
    viewport: {
      width: 1440,
      height: 2000,
    },
  });
  await context.addInitScript(() => {
    Object.defineProperty(navigator, "webdriver", {
      get: () => undefined,
    });
  });
  const page = await context.newPage();

  await page.goto(inputUrl, {
    timeout: 60000,
    waitUntil: "domcontentloaded",
  });

  try {
    await page.waitForLoadState("networkidle", {
      timeout: 10000,
    });
  } catch {
    // Some pages never reach networkidle; best-effort is enough here.
  }
  await page.waitForTimeout(2000);

  const finalUrl = page.url() || inputUrl;
  const pageTitle = (await page.title()) || "";
  const html = await page.content();
  const dom = new JSDOM(html, { url: finalUrl });
  const result = await Defuddle(dom.window.document, finalUrl, {
    markdown: true,
    useAsync: false,
  });

  const metadata = {};
  const metadataEntries = {
    author: result.author,
    description: result.description,
    domain: result.domain,
    image: result.image,
    language: result.language,
    published: result.published,
    site: result.site,
    title: result.title || pageTitle,
  };

  for (const [key, value] of Object.entries(metadataEntries)) {
    if (!value) {
      continue;
    }
    metadata[key] = value;
  }

  const payload = {
    content: result.content || "",
    final_url: finalUrl,
    metadata,
    title: result.title || pageTitle,
  };

  console.log(JSON.stringify(payload));
} catch (error) {
  const errorMessage = error instanceof Error ? error.message : String(error);
  console.error(errorMessage);
  process.exit(1);
} finally {
  if (browser) {
    await browser.close();
  }
}
