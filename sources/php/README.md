# `php` — PHP & Ecosystem

PHP language reference, best practices, framework documentation, testing frameworks,
dependency management, PSR standards, and design patterns.

## What belongs here

- **PHP language reference** — official PHP manual, stdlib docs
- **PHP best practices** — modern idioms, PHP 8+ features, community guides
- **Framework documentation** — Laravel, CodeIgniter, Symfony, Slim, or any PHP framework
  in use by your project (any version)
- **Testing frameworks** — PHPUnit, Pest, Codeception, or other PHP test tooling
- **Dependency management** — Composer docs, Packagist conventions
- **PSR standards** — PHP-FIG accepted PSRs (coding style, autoloading, HTTP interfaces,
  logging, caching, DI, event dispatching)
- **Design patterns** — GoF patterns implemented in PHP, architectural patterns
- **Clean code / refactoring** — PHP-specific clean code and refactoring guides

## Suggested sources

The following are high-quality, freely available resources suitable for ingestion:

| Source | Where to get it | Notes |
|---|---|---|
| PHP Manual | [php.net/download-docs.php](https://www.php.net/download-docs.php) | Single HTML or many-HTML archive |
| PHP: The Right Way | [phptherightway.com](https://phptherightway.com) or [GitHub](https://github.com/codeguy/php-the-right-way) | Community best practices |
| Laravel docs | [github.com/laravel/docs](https://github.com/laravel/docs) | Branch per version (e.g. `12.x`, `11.x`) |
| CodeIgniter docs | [github.com/bcit-ci/CodeIgniter](https://github.com/bcit-ci/CodeIgniter) `user_guide/` | Tag per version |
| PHPUnit docs | [github.com/sebastianbergmann/phpunit-documentation-english](https://github.com/sebastianbergmann/phpunit-documentation-english) | Branch per version; RST format |
| Pest docs | [github.com/pestphp/docs](https://github.com/pestphp/docs) | Branch per version (e.g. `4.x`) |
| Composer docs | [github.com/composer/composer](https://github.com/composer/composer) `doc/` | Branch `main` |
| PHP-FIG PSRs | [github.com/php-fig/fig-standards](https://github.com/php-fig/fig-standards) `accepted/` | All accepted PSRs in Markdown |
| DesignPatternsPHP | [github.com/DesignPatternsPHP/DesignPatternsPHP](https://github.com/DesignPatternsPHP/DesignPatternsPHP) | RST + PHP 8 examples |
| Clean Code PHP | [github.com/jupeter/clean-code-php](https://github.com/jupeter/clean-code-php) | Single README; Clean Code adapted for PHP |

## Coverage areas

When populating this collection, aim to cover:

### Language & Standards
- Core language syntax, types, OOP, fibers, match expressions, named arguments (PHP 8+)
- SPL, standard functions, and built-in interfaces
- PSR-1/2/12 coding style, PSR-3 logging, PSR-4 autoloading
- PSR-6/16 caching, PSR-7/17/18 HTTP, PSR-11 container, PSR-14 events

### Frameworks
- Routing, controllers, middleware, request/response lifecycle
- ORM / query builder / database abstraction
- Templating, authentication, authorization
- CLI tooling, queues, events, broadcasting

### Testing
- Writing unit, integration, and feature tests
- Assertions, test doubles (stubs, mocks, spies)
- Data providers / datasets
- Code coverage configuration and reporting
- Architecture testing (dependency and layer rules)

### Tooling
- Composer: `composer.json` schema, CLI commands, autoloading, scripts, version constraints
- CI integration for test suites
- Static analysis (PHPStan, Psalm) if relevant to your project

## Format notes

- Markdown (`.md`) and reStructuredText (`.rst`) are both supported by the ingestion pipeline
- Single-file HTML (e.g. the PHP manual) and multi-file HTML are supported
- Prefer sparse clones of large repositories to avoid ingesting unrelated files
  (e.g. clone only the `docs/` or `user_guide/` subdirectory)
- Remove `.git` directories before ingesting to keep the collection clean
