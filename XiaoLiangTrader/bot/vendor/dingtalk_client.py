import json
import logging
from typing import Dict, List, Any, Optional
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger("dingtalk_ai_table_client")

class DingTalkAITableClient:
    """
    DingTalk AI Table (Multi-dimensional Table) MCP Client Wrapper.
    Simplifies interactions with the DingTalk AI Table MCP Server using standard python methods.
    """
    def __init__(self, server_url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0):
        self.server_url = server_url
        self.headers = headers or {}
        self.timeout = timeout
        
        self._http_client = None
        self._client_ctx = None
        self._read_stream = None
        self._write_stream = None
        self.session: Optional[ClientSession] = None

    async def __aenter__(self):
        """Asynchronously connect to the Streamable HTTP endpoint and initialize MCP session."""
        logger.info(f"Connecting to DingTalk AI Table MCP Server at: {self.server_url}")
        
        self._http_client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout)
        await self._http_client.__aenter__()
        
        self._client_ctx = streamable_http_client(
            url=self.server_url, 
            http_client=self._http_client
        )
        self._read_stream, self._write_stream, _ = await self._client_ctx.__aenter__()
        
        self.session = ClientSession(self._read_stream, self._write_stream)
        await self.session.__aenter__()
        
        logger.info("Initializing MCP Session...")
        await self.session.initialize()
        logger.info("MCP Session initialized successfully.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session and connections."""
        logger.info("Closing MCP Client Session...")
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client_ctx:
            await self._client_ctx.__aexit__(exc_type, exc_val, exc_tb)
        if self._http_client:
            await self._http_client.__aexit__(exc_type, exc_val, exc_tb)
        logger.info("Connection closed.")

    def _parse_response(self, result) -> Any:
        """
        Parses the JSON response from the MCP server.
        DingTalk AI Table returns JSON formatted strings inside TextContent.
        """
        if not result or not hasattr(result, "content"):
            return result
        
        texts = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
                
        if not texts:
            return None
            
        full_text = "".join(texts)
        try:
            return json.loads(full_text)
        except json.JSONDecodeError:
            return full_text

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Generic method to call any tool on the MCP server."""
        if not self.session:
            raise RuntimeError("Client is not connected. Use 'async with' context manager.")
        
        # In DingTalk AI Table, tool names can sometimes be accessed with or without dot prefix.
        # Here we normalize the name to strip leading dot if the session expects it clean, 
        # or we keep it. The official tools list is list_bases, create_base etc.
        args = arguments or {}
        logger.debug(f"Calling tool '{tool_name}' with args: {args}")
        raw_result = await self.session.call_tool(tool_name, args)
        return self._parse_response(raw_result)

    # ==========================================
    # Base Management APIs
    # ==========================================
    async def list_bases(self, limit: int = 10, cursor: Optional[str] = None) -> Dict[str, Any]:
        """List all accessible bases."""
        args = {"limit": limit}
        if cursor:
            args["cursor"] = cursor
        return await self.call_tool("list_bases", args)

    async def search_bases(self, query: str, cursor: Optional[str] = None) -> Dict[str, Any]:
        """Search bases by name."""
        args = {"query": query}
        if cursor:
            args["cursor"] = cursor
        return await self.call_tool("search_bases", args)

    async def get_base(self, base_id: str) -> Dict[str, Any]:
        """Get directory information (tables / dashboards summary) for a base."""
        return await self.call_tool("get_base", {"baseId": base_id})

    async def create_base(self, base_name: str, template_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Base."""
        args = {"baseName": base_name}
        if template_id:
            args["templateId"] = template_id
        return await self.call_tool("create_base", args)

    async def update_base(self, base_id: str, new_base_name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Update Base name and/or description."""
        args = {"baseId": base_id, "newBaseName": new_base_name}
        if description:
            args["description"] = description
        return await self.call_tool("update_base", args)

    async def delete_base(self, base_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Delete a Base (High Risk)."""
        args = {"baseId": base_id}
        if reason:
            args["reason"] = reason
        return await self.call_tool("delete_base", args)

    async def search_templates(self, query: str, limit: int = 10, cursor: Optional[str] = None) -> Dict[str, Any]:
        """Search base templates."""
        args = {"query": query, "limit": limit}
        if cursor:
            args["cursor"] = cursor
        return await self.call_tool("search_templates", args)

    # ==========================================
    # Table Management APIs
    # ==========================================
    async def get_tables(self, base_id: str, table_ids: List[str]) -> Dict[str, Any]:
        """Batch retrieve table schemas and fields summary."""
        return await self.call_tool("get_tables", {"baseId": base_id, "tableIds": table_ids})

    async def create_table(self, base_id: str, table_name: str, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a new Table with initial fields (up to 15 fields).
        fields format: [{"fieldName": "Name", "type": "text"}, ...]
        """
        args = {"baseId": base_id, "tableName": table_name, "fields": fields}
        return await self.call_tool("create_table", args)

    async def update_table(self, base_id: str, table_id: str, new_table_name: str) -> Dict[str, Any]:
        """Rename a Table."""
        args = {"baseId": base_id, "tableId": table_id, "newTableName": new_table_name}
        return await self.call_tool("update_table", args)

    async def delete_table(self, base_id: str, table_id: str) -> Dict[str, Any]:
        """Delete a Table."""
        args = {"baseId": base_id, "tableId": table_id}
        return await self.call_tool("delete_table", args)

    # ==========================================
    # Field Management APIs
    # ==========================================
    async def get_fields(self, base_id: str, table_id: str, field_ids: List[str]) -> Dict[str, Any]:
        """Get full details of specified fields."""
        return await self.call_tool("get_fields", {"baseId": base_id, "tableId": table_id, "fieldIds": field_ids})

    async def create_fields(self, base_id: str, table_id: str, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch create fields in a table."""
        args = {"baseId": base_id, "tableId": table_id, "fields": fields}
        return await self.call_tool("create_fields", args)

    async def update_field(self, base_id: str, table_id: str, field_id: str, new_field_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update a field's name or configuration."""
        args = {"baseId": base_id, "tableId": table_id, "fieldId": field_id}
        if new_field_name:
            args["newFieldName"] = new_field_name
        if config:
            args["config"] = config
        return await self.call_tool("update_field", args)

    async def delete_field(self, base_id: str, table_id: str, field_id: str) -> Dict[str, Any]:
        """Delete a field from a table."""
        args = {"baseId": base_id, "tableId": table_id, "fieldId": field_id}
        return await self.call_tool("delete_field", args)

    # ==========================================
    # Record Management APIs
    # ==========================================
    async def query_records(
        self, 
        base_id: str, 
        table_id: str, 
        limit: int = 100, 
        record_ids: Optional[List[str]] = None,
        search_word: Optional[str] = None,
        filter_query: Optional[str] = None,
        sort: Optional[List[Dict[str, Any]]] = None,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query records in a table.
        Allows sorting, searching, and filtering.
        """
        args = {
            "baseId": base_id,
            "tableId": table_id,
            "limit": limit
        }
        if record_ids:
            args["recordIds"] = record_ids
        if search_word:
            args["searchWord"] = search_word
        if filter_query:
            args["filterQuery"] = filter_query
        if sort:
            args["sort"] = sort
        if cursor:
            args["cursor"] = cursor
            
        return await self.call_tool("query_records", args)

    async def create_records(self, base_id: str, table_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch create records.
        records format: [{"cells": {"fld_xxx": "Value"}}, ...]
        """
        args = {"baseId": base_id, "tableId": table_id, "records": records}
        return await self.call_tool("create_records", args)

    async def update_records(self, base_id: str, table_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch update records.
        records format: [{"recordId": "rec_xxx", "cells": {"fld_xxx": "New Value"}}, ...]
        """
        args = {"baseId": base_id, "tableId": table_id, "records": records}
        return await self.call_tool("update_records", args)

    async def delete_records(self, base_id: str, table_id: str, record_ids: List[str]) -> Dict[str, Any]:
        """Batch delete records."""
        args = {"baseId": base_id, "tableId": table_id, "recordIds": record_ids}
        return await self.call_tool("delete_records", args)

    # ==========================================
    # Attachment Upload API
    # ==========================================
    async def prepare_attachment_upload(self, base_id: str, table_id: str, file_name: str, file_size: int) -> Dict[str, Any]:
        """Request OSS upload authorization for attachment field."""
        args = {
            "baseId": base_id,
            "tableId": table_id,
            "fileName": file_name,
            "fileSize": file_size
        }
        return await self.call_tool("prepare_attachment_upload", args)
