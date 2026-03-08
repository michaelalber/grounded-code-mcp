# `javascript_typescript` — JavaScript & TypeScript

JavaScript language reference, modern idioms, TypeScript, and jQuery for
legacy application maintenance. Covers vanilla JS, ES2024 spec, and both
legacy jQuery patterns and modern TypeScript.

## What belongs here

- JavaScript language reference and best practices
- TypeScript handbook and practical guides
- jQuery API and learning resources (legacy app support)
- ECMAScript specification (precise semantics)

## Sources in this collection

### Books (local)
| File | Content |
|---|---|
| `javascript-the-definitive-guide-flanagan.pdf` | JS: The Definitive Guide — Flanagan (O'Reilly) |
| `murach-javascript-and-jquery.pdf` | Murach JavaScript and jQuery |
| `secrets-of-the-javascript-ninja.pdf` | Secrets of the JavaScript Ninja — Resig & Bibeault |
| `modern-javascript-develop-and-design.pdf` | Modern JavaScript: Develop and Design — Ullman |
| `programming-javascript-applications-elliott.epub` | Programming JavaScript Applications — Eric Elliott |
| `oop-in-javascript.epub` | Principles of OOP in JavaScript |

### Free/Open Resources (downloaded)
| Directory / File | Content |
|---|---|
| `eloquent-javascript.pdf` | Eloquent JavaScript 3rd ed. — Haverbeke (CC BY-NC) |
| `ecmascript-2024-spec.pdf` | ECMAScript 2024 (ECMA-262 15th ed.) official spec |
| `you-dont-know-js-2nd-ed/` | You Don't Know JS 2nd ed. — Kyle Simpson (GitHub) |
| `javascript-info/` | javascript.info — Modern JS Tutorial (712 articles) |
| `typescript-handbook/` | TypeScript Handbook — official Microsoft source |
| `typescript-deep-dive/` | TypeScript Deep Dive — Basarat Ali Syed |
| `jquery-learning-center/` | jQuery Learning Center — jquery.com (94 guides) |

## Refreshing open-source content

```bash
JS_DST=sources/javascript

# Eloquent JavaScript (updated per edition)
curl -sL https://eloquentjavascript.net/Eloquent_JavaScript.pdf -o $JS_DST/eloquent-javascript.pdf

# ECMAScript spec (annual release, update year as needed)
curl -sL https://www.ecma-international.org/wp-content/uploads/ECMA-262_15th_edition_june_2024.pdf \
     -o $JS_DST/ecmascript-2024-spec.pdf

# You Don't Know JS
curl -sL https://github.com/getify/You-Dont-Know-JS/archive/refs/heads/2nd-ed.zip | \
  funzip | tar -x && mv You-Dont-Know-JS-2nd-ed $JS_DST/you-dont-know-js-2nd-ed

# JavaScript.info
curl -sL https://github.com/javascript-tutorial/en.javascript.info/archive/refs/heads/master.zip | \
  funzip | tar -x && mv en.javascript.info-master $JS_DST/javascript-info

# TypeScript Deep Dive
curl -sL https://github.com/basarat/typescript-book/archive/refs/heads/master.zip | \
  funzip | tar -x && mv typescript-book-master $JS_DST/typescript-deep-dive

# TypeScript Handbook (sparse checkout)
git clone --depth=1 --filter=blob:none --sparse https://github.com/microsoft/TypeScript-Website.git /tmp/ts-website
cd /tmp/ts-website && git sparse-checkout set packages/documentation/copy/en
cp -r packages/documentation/copy/en $JS_DST/typescript-handbook && rm -rf /tmp/ts-website

# jQuery Learning Center
curl -sL https://github.com/jquery/learn.jquery.com/archive/refs/heads/main.zip | \
  funzip | tar -x && mv learn.jquery.com-main $JS_DST/jquery-learning-center
```
