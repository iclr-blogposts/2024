#!/usr/bin/env python3

import re
import sys

SUCCESS = True

SLUG = sys.argv[1]

OUTPUT_MSG = ""

SLUG_TEMPLATE = "2024-\d\d-\d\d-.+"
if re.match(SLUG_TEMPLATE, SLUG) is None:
    print("Your slug does not match the template! Please change it.")
    print(f"Your slug: {SLUG}")
    print(f"The template: {SLUG_TEMPLATE}")
    print("PATHFILTERFAILED")
    SUCCESS = False
    OUTPUT_MSG = f"Your PR title does not match the slug template, which is <{SLUG_TEMPLATE}>."

CHANGED_FILES = sys.argv[2:]
ACCEPTABLE_PATHS = [
    f"_posts/{SLUG}.md",
    f"assets/img/{SLUG}/*",
    f"assets/html/{SLUG}/*",
    f"assets/bibliography/{SLUG}.bib"
]

failed_paths = []

for changed_file in CHANGED_FILES:
    for acc_path in ACCEPTABLE_PATHS:
        if re.match(acc_path, changed_file) is not None:
            break
    else:
        failed_paths.append(changed_file)

if len(failed_paths) > 0:
    print(f"These files were changed, but they shouldn't have been:")
    for failed in failed_paths:
        print(f"\t{failed}")

    print("PATHFILTERFAILED")
    SUCCESS = False
else:
    print("PATHFILTERSUCCESS")
    SUCCESS = True

if len(failed_paths) > 0:
    if OUTPUT_MSG != "":
        OUTPUT_MSG += " Also, y"
    else:
        OUTPUT_MSG = "Y"
    
    OUTPUT_MSG += f"ou can only add/change/remove files related to your post, i.e. files that match one of these patterns: <_posts/SLUG.md, assets/img/SLUG/..., assets/html/SLUG/..., assets/bibliography/SLUG.bib>. But we found that you changed the following: <{' & '.join(failed_paths)}>"
if not SUCCESS:
    OUTPUT_MSG += " Also, make sure your PR's title matches your post's slug!"
    print(OUTPUT_MSG)

# example usage of this script:  python3 filter_file.py 2024-0a1-01-whateve _posts/2024-01-01-whateve.md assets/img/2024-01-01-whateve/bla.pic assets/html/2024-01-01-whateve/plot1.j assets/bibliography/2024-01-01-whateve.bib assets/img/2024-01-02-whateve/bla.pic
if SUCCESS:
    exit(0)
else:
    exit(1)
