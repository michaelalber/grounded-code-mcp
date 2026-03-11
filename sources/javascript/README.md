# `javascript` — JavaScript, TypeScript & CSS Conventions

JavaScript language reference, modern idioms, TypeScript, Vue.js, jQuery, and
CSS naming conventions. Covers vanilla JS, the ECMAScript specification, legacy
jQuery patterns, modern TypeScript, Vue 2/3 frameworks, and component-based CSS
methodologies such as BEM.

## What belongs here

- **JavaScript language reference** — core language, built-ins, the event loop, modules
- **TypeScript** — handbook, type system, compiler options, declaration files
- **ECMAScript specification** — precise semantics, proposal tracking
- **Frameworks & libraries** — Vue.js (2 and 3), jQuery, or any JS/TS framework in use
- **Coding standards & style guides** — Google TypeScript Style Guide (supersedes the JS guide), Airbnb, StandardJS, or similar
- **CSS naming conventions** — BEM, OOCSS, SMACSS, SUITCSS, or similar methodology docs
- **Books and deep-dives** — language internals, functional patterns, asynchronous programming

## Suggested sources

The following are high-quality, freely available resources suitable for ingestion:

| Source | Where to get it | Notes |
|---|---|---|
| ECMAScript spec | [tc39.es/ecma262](https://tc39.es/ecma262/) or PDF release | Current living standard |
| MDN Web Docs (JS) | [github.com/mdn/content](https://github.com/mdn/content) `files/en-us/web/javascript/` | Comprehensive reference; large repo — sparse clone |
| Google TypeScript Style Guide | [google.github.io/styleguide/tsguide.html](https://google.github.io/styleguide/tsguide.html) | Single-page HTML; convert to Markdown; supersedes the JS guide |
| Google JavaScript Style Guide | [google.github.io/styleguide/jsguide.html](https://google.github.io/styleguide/jsguide.html) | **Deprecated** — Google recommends migrating to TypeScript |
| TypeScript Handbook | [github.com/microsoft/TypeScript-Website](https://github.com/microsoft/TypeScript-Website) `packages/documentation/` | Official handbook source in Markdown |
| TypeScript Deep Dive | [github.com/basarat/typescript-book](https://github.com/basarat/typescript-book) | Community deep-dive; Markdown |
| You Don't Know JS | [github.com/getify/You-Dont-Know-JS](https://github.com/getify/You-Dont-Know-JS) | 2nd ed; Markdown; all 6 books |
| javascript.info | [github.com/javascript-tutorial/en.javascript.info](https://github.com/javascript-tutorial/en.javascript.info) | Full modern JS tutorial; Markdown |
| Eloquent JavaScript | [eloquentjavascript.net](https://eloquentjavascript.net) or [GitHub](https://github.com/marijnh/Eloquent-JavaScript) | Free book; HTML and PDF |
| Vue.js 3 docs | [github.com/vuejs/docs](https://github.com/vuejs/docs) | Official docs; Markdown |
| Vue.js 2 docs | [github.com/vuejs/v2.vuejs.org](https://github.com/vuejs/v2.vuejs.org) | Legacy v2 docs |
| jQuery Learning Center | [github.com/jquery/learn.jquery.com](https://github.com/jquery/learn.jquery.com) | Markdown; jQuery patterns |
| BEM Methodology | [github.com/bem-site/bem-method](https://github.com/bem-site/bem-method) `method/` | Official BEM docs; Markdown; branch `bem-info-data` |

## Coverage areas

When populating this collection, aim to cover:

### Language & Specification
- Core syntax, types, prototypes, closures, generators, async/await, modules
- Event loop, microtask queue, memory model
- ECMAScript proposals and version history (ES2015 through current)
- Strict mode, coercion, equality, and common footguns

### TypeScript
- Type system fundamentals: unions, intersections, generics, conditional types
- Compiler options (`tsconfig.json`), declaration files (`.d.ts`)
- Type narrowing, discriminated unions, template literal types
- Integration with bundlers and frameworks

### Frameworks & Libraries
- Component lifecycle, reactivity, state management
- Routing, forms, composables/hooks
- jQuery: selectors, DOM manipulation, AJAX, event delegation (for legacy codebases)

### Coding Standards & Style
- Naming conventions, formatting rules, import/export style
- Linting configuration (ESLint, Prettier) and rule rationale
- Module system conventions (CommonJS vs. ESM)

### CSS Naming Conventions
- BEM: Block, Element, Modifier — naming rules, alternative schemes, CSS application
- Related methodologies: OOCSS, SMACSS, SUITCSS for comparison
- File structure conventions for component-based CSS

## Format notes

- Markdown (`.md`) is the preferred format; HTML is also supported
- For large repositories, use a sparse clone to fetch only the relevant subdirectory
  (e.g. `docs/`, `method/`, or `packages/documentation/`)
- Single-page HTML guides (e.g. Google style guides) can be converted to Markdown
  using `pandoc` or a simple HTML-to-Markdown script
- Remove `.git` directories before ingesting to keep the collection clean
