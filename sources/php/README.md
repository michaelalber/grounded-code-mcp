# `php_laravel` — PHP & Laravel

PHP language reference, best practices, and Laravel framework documentation.
Covers legacy Laravel versions (5.5, 6.x) for migration projects alongside
the current stable release.

## What belongs here

- PHP language reference and standard library docs
- PHP best practices and modern idioms (PHP 8+)
- Laravel framework documentation (multiple versions)
- Composer, testing with PHPUnit/Pest, and ecosystem tools

## Sources in this collection

| File / Directory | Content |
|---|---|
| `php_manual_en.html` | Full PHP manual (downloaded from php.net) |
| `php-the-right-way.md` | PHP: The Right Way — best practices guide |
| `laravel-5.5/` | Laravel 5.5 docs (GitHub: laravel/docs@5.5) |
| `laravel-6x/` | Laravel 6.x LTS docs (GitHub: laravel/docs@6.x) |
| `laravel-12x/` | Laravel 12.x docs — current stable (GitHub: laravel/docs@12.x) |

## Refreshing sources

```bash
# PHP manual (updated daily at php.net)
curl -L https://php.net/distributions/manual/php_manual_en.html.gz | gunzip > sources/php/php_manual_en.html

# Laravel docs (re-download to pick up updates)
curl -sL https://github.com/laravel/docs/archive/refs/heads/12.x.zip -o /tmp/lara.zip
unzip -q /tmp/lara.zip -d /tmp && rm -rf sources/php/laravel-12x && mv /tmp/docs-12.x sources/php/laravel-12x

# PHP: The Right Way
git clone --depth=1 https://github.com/codeguy/php-the-right-way.git /tmp/ptrtw
# concatenate _posts/*.md in order into sources/php/php-the-right-way.md
```
