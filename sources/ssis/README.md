# SSIS Sources

SQL Server Integration Services (SSIS) documentation for the `grounded_ssis` collection.

## Collection

Pass `collection="ssis"` in MCP tool calls to search this collection.

## What Lives Here

| Path | Source | Coverage |
|------|--------|----------|
| `microsoft-learn/` | learn.microsoft.com/en-us/sql/integration-services | Official Microsoft SSIS docs |

### Microsoft Learn Coverage

The official SSIS documentation covers:

- **Overview** — architecture, packages, components, SSIS Designer
- **Quickstarts** — deploy and run packages via SSMS, T-SQL, PowerShell, C#
- **Control Flow** — tasks (Execute SQL, Script, Data Flow, File System, FTP, Send Mail, Execute Package, Bulk Insert, XML, WMI), containers (For Loop, Foreach Loop, Sequence), precedence constraints
- **Data Flow** — sources (OLE DB, Flat File, Excel, XML, ADO.NET, ODBC, SAP BW), destinations (OLE DB, Flat File, Excel, ADO.NET, ODBC, SQL Server, Data Mining), transformations (Lookup, Aggregate, Sort, Derived Column, Conditional Split, Data Conversion, Merge, Merge Join, Union All, Multicast, Pivot/Unpivot, Slowly Changing Dimension, Fuzzy Lookup/Grouping, Term Extraction, Character Map, Audit, Row Count)
- **Connection Managers** — OLE DB, ADO.NET, Flat File, Excel, ODBC, FTP, HTTP, SMTP, MSMQ, WMI, Analysis Services
- **Variables and Parameters** — scope, data types, expressions
- **SSIS Expressions** — operators, functions, data type casting, expression language reference
- **Event Handlers** — package-level and container-level events
- **Logging** — log providers, custom logging, SSIS Catalog log views
- **Package Deployment** — project deployment model, package deployment model, ISDeploymentWizard
- **SSIS Catalog** — SSISDB, stored procedures, views, operations, reports
- **Performance** — data flow optimization, buffer configuration, row counts, profiling
- **Scripting** — Script Task (C#/VB.NET), Script Component (source/transformation/destination)
- **Azure** — Azure-enabled SSIS, Azure-SSIS Integration Runtime in ADF

## Downloading

```bash
python download_ssis_docs.py
```

Requires: `pip install requests beautifulsoup4 html2text`

## Ingesting

```bash
grounded-code-mcp ingest sources/ssis --collection ssis
```

## Additional sources

PDFs of SSIS books or supplementary references can be placed under `sources/ssis/books/`
and will be ingested automatically.
