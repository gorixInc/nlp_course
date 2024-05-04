import asyncio
from pyppeteer import launch
import os
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from slugify import slugify
import os

async def parse_content_page(page, category, section):
    content_selector = '.accordion'
    title_selector = '.card h4'
    await page.waitForSelector(content_selector, options={'visible': True, 'timeout': 3000})
    content = await page.content()

    soup = BeautifulSoup(content, 'html.parser')
    title = slugify(soup.select_one(title_selector).text)
    print(title)
    panels = soup.select(".accordion .accordion__item")[:-1]
    whole_accordion = ''.join([str(panel) for panel in panels])
    strip_elements = []

    output = md(
        str(whole_accordion),
        keep_inline_images_in=['td', 'th', 'a', 'figure'],
        strip=strip_elements
    )
    if len(output) == 0:
        return 
    
    file_path = f'{category}/{section}/{title}.md'
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(output)

async def click_button(page, selector, ind):  
    await page.waitForSelector(selector, options={'visible': True, 'timeout': 3000})
    butts = await page.querySelectorAll(selector)
    await butts[ind].click()

async def parse_category(category_name, category_link):
    side_b_sel = '#sidemenu-tags .card--link' 
    arrow_b_sel = '.arrow-list .arrow-list__item' 
    section_selector = '.grid__col h2'

    os.makedirs(f'{category_name}', exist_ok=True)

    browser = await launch()
    page = await browser.newPage()
    await page.goto(category_link)
    await page.waitForSelector('#allowAll', options={'visible': True})
    # Click the button using its ID
    await page.click('#allowAll')
    await page.waitFor(500)

    await page.waitForSelector(side_b_sel, options={'visible': True})
    side_butts = await page.querySelectorAll(side_b_sel)

    for side_ind in range(len(side_butts)):

        try:
            await click_button(page, side_b_sel, side_ind)
            await page.waitFor(500)  # 500 ms delay
        except:
            continue
        try:
            await page.waitForSelector(arrow_b_sel, options={'visible': True,'timeout': 3000})
            arrow_butts = await page.querySelectorAll(arrow_b_sel)
        except:
            continue

        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        section = slugify(soup.select_one(section_selector).text)
        os.makedirs(f'{category_name}/{section}', exist_ok=True)

        for arrow_ind in range(len(arrow_butts)):
            clicked_button = False
            try:
                await click_button(page, arrow_b_sel, arrow_ind)
                clicked_button = True
                await page.waitFor(1000)
                await parse_content_page(page, category_name, section)
            except Exception as e:
                print(f'Failed {section, category_name} here ^^^^^^^^^^^^')
                print(e)

            if clicked_button:
                await page.goBack()
    await browser.close()

#asyncio.get_event_loop().run_until_complete(parse_category('arveldus', 'https://www.telia.ee/abi/teema/641/arveldus'))
#asyncio.get_event_loop().run_until_complete(parse_category('mobiil', 'https://www.telia.ee/abi/teema/3/mobiil'))
#asyncio.get_event_loop().run_until_complete(parse_category('tv', 'https://www.telia.ee/abi/teema/2/tv'))
#asyncio.get_event_loop().run_until_complete(parse_category('it_lahendused', 'https://www.telia.ee/abi/teema/774/it-lahendused'))
#asyncio.get_event_loop().run_until_complete(parse_category('e-pood', 'https://www.telia.ee/abi/teema/788/e-pood'))
asyncio.get_event_loop().run_until_complete(parse_category('opi-ja-opita', 'https://www.telia.ee/abi/teema/1288/õpi-ja-õpeta'))