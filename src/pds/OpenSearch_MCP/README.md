# PDS4 Semantic Search with OpenSearch MCP

Search Planetary Data System (PDS4) metadata using natural language through Claude Desktop and the OpenSearch Model Context Protocol (MCP) server.

![OpenSearch Standalone MCP server architecture](https://opensearch.org/wp-content/uploads/2025/05/standalone-mcp-architecture-1024x404.png)
Source: https://opensearch.org/blog/introducing-mcp-in-opensearch/

## What This Does

Ask questions about planetary science data in plain English, and get:
1. **Relevant PDS4 datasets** from our OpenSearch collection
2. **Custom analysis scripts** for downloading and analyzing the data
3. **Ready-to-run code** for visualization and data processing

## Example Workflow

**You ask:** "Find Mars rover thermal data from 2021"

**System provides:**
- Matching PDS4 collections from the `en-registry` index  
- Python script to download the thermal data
- Visualization code to create temperature maps
- Analysis tools for the specific dataset

## Quick Start

### 1. Prerequisites
- Access to PDS OpenSearch cluster with `en-registry` index
- Claude Desktop installed
- Python environment for running generated scripts

### 2. Add to Claude Desktop

Add this to your Claude Desktop MCP configuration:

```json
{
    "mcpServers": {
    "opensearch-mcp-server": {
      "command": "python3",
      "args": ["-m","mcp_server_opensearch"],
      "env": {
          "OPENSEARCH_URL": "...",
          "AWS_REGION": "us-west-2",
          "AWS_ACCESS_KEY_ID": "...",
          "AWS_SECRET_ACCESS_KEY": "...",
          "AWS_SESSION_TOKEN": "...",
          "AWS_OPENSEARCH_SERVERLESS": "true"
      }
  }
}}
```

- `OPENSEARCH_URL` - OpenSearch collection serverless URL (found in AWS console)
- `AWS_OPENSEARCH_SERVERLESS` - set to True if serverless collection
- `AWS_**` - AWS IAM credentials

More info on OpenSearch MCP Server [here](https://github.com/opensearch-project/opensearch-mcp-server-py).

### 3. Start Searching

Open Claude Desktop and try queries like:
- "Show me Europa ice data from recent missions"
- "Find atmospheric measurements from Venus Express"
- "I need spectral data of Martian minerals"

## What You Get

The system generates scripts that can:
- Download PDS4 data automatically
- Handle planetary science data formats
- Create visualizations (maps, plots, time series)
- Perform analysis (statistics, spectral analysis, comparisons)
- Export results in various formats

## Example Scripts

See the included Jupyter notebook for real examples of:
- Natural language queries
- Generated download scripts
- Data visualization code
- Analysis workflows for PDS4 datasets

## About OpenSearch MCP

This project uses the OpenSearch Model Context Protocol integration. For more details about OpenSearch MCP capabilities, see: [Introducing MCP in OpenSearch](https://opensearch.org/blog/introducing-mcp-in-opensearch/) and its [GitHub](https://github.com/opensearch-project/opensearch-mcp-server-py).

## Data Coverage

Our `en-registry` index includes PDS4 metadata for:
- Mars rover missions (Curiosity, Perseverance, etc.)
- Orbital missions (MRO, MAVEN, etc.)
- Outer planet missions (Juno, Cassini, etc.)
- Laboratory and Earth-based observations
- Historical mission archives

## Need Help?

**Common issues:**
- **No connection**: Check your OpenSearch host and credentials
- **No results**: Verify the `en-registry` index is accessible
- **Script errors**: Make sure PDS4 metadata fields are properly indexed

## Contributing

Found a bug or want to add features? Contributions welcome!

---

*This tool makes PDS4 data discovery as easy as asking a question. No more manual catalog searches or complex query syntax - just describe what you need in natural language.*
