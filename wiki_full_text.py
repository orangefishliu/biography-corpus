import wikipediaapi
import json

# load wikipedia extractor
wiki_extract = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)


# a function to get a list of wikipedia's category members
def list_category_members(category_members):
    cat_list = []
    for c in category_members.values():
        cat_list.append(c.title)
    return cat_list


# a list to put all page contents of category members
ppl_bio = []
# get all members' names from Category:Leaders of political parties in Germany
cat = wiki_extract.page("Category:Leaders of political parties in Germany")
ppl_list = list_category_members(cat.categorymembers)

# iterate each member to get their page contents
for person in ppl_list:
    page_ppl = wiki_extract.page(person)
    bio_example = (page_ppl.title, page_ppl.text)
    ppl_bio.append(bio_example)

# write all members' page contents to a json file
json_string = json.dumps(ppl_bio, indent=4, ensure_ascii=False).encode("utf-8")
with open("ppl_bio.json", "w", encoding="utf-8") as outfile:
    outfile.write(json_string.decode())

