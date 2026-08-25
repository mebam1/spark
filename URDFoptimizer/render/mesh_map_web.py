#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless HTTP web app for mapping split GLB meshes to LLM URDF links."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlparse

try:
    from URDFoptimizer.render.mesh_map_gui import (
        UNASSIGNED,
        _build_mesh_map,
        _load_existing_assignments,
        _load_json,
        _load_split_parts,
        _metadata_links,
        _save_json,
    )
except ImportError:
    from mesh_map_gui import (  # type: ignore
        UNASSIGNED,
        _build_mesh_map,
        _load_existing_assignments,
        _load_json,
        _load_split_parts,
        _metadata_links,
        _save_json,
    )


DEFAULT_VIEWER_SCRIPT = "https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"


class MeshMapWebState:
    def __init__(
        self,
        metadata_path: Path,
        split_dir: Path,
        output_path: Path,
        absolute_paths: bool = False,
        viewer_script_url: str = DEFAULT_VIEWER_SCRIPT,
    ) -> None:
        self.metadata_path = metadata_path
        self.split_dir = split_dir
        self.output_path = output_path
        self.absolute_paths = absolute_paths
        self.viewer_script_url = viewer_script_url

        self.metadata = _load_json(metadata_path)
        self.links = _metadata_links(self.metadata)
        self.link_labels = [link["label"] for link in self.links]
        self.split_parts = _load_split_parts(split_dir)
        self.parts_by_file = {part["file"]: part for part in self.split_parts}

        if not self.links:
            raise ValueError(f"No parts found in metadata: {metadata_path}")
        if not self.split_parts:
            raise ValueError(f"No split GLB files found in: {split_dir}")

    def assignments(self) -> Dict[str, str]:
        existing = _load_existing_assignments(self.output_path, self.split_parts)
        return {part["file"]: existing.get(part["file"], UNASSIGNED) for part in self.split_parts}

    def to_json(self) -> Dict[str, Any]:
        return {
            "object_name": self.metadata.get("object_name", "object"),
            "metadata": str(self.metadata_path),
            "split_dir": str(self.split_dir),
            "output": str(self.output_path),
            "unassigned": UNASSIGNED,
            "links": self.links,
            "parts": [
                {
                    "index": part.get("index"),
                    "file": part["file"],
                    "node_name": part.get("node_name", ""),
                    "geometry_name": part.get("geometry_name", ""),
                    "vertices": part.get("vertices", ""),
                    "faces": part.get("faces", ""),
                    "mesh_url": f"/mesh/{quote(part['file'])}",
                }
                for part in self.split_parts
            ],
            "assignments": self.assignments(),
        }

    def save_assignments(self, raw_assignments: Dict[str, Any]) -> Dict[str, Any]:
        valid_links = set(self.link_labels)
        assignments: List[str] = []
        for part in self.split_parts:
            link = str(raw_assignments.get(part["file"], UNASSIGNED))
            if link not in valid_links:
                link = UNASSIGNED
            assignments.append(link)

        mesh_map = _build_mesh_map(self.split_parts, assignments, self.output_path, self.absolute_paths)
        _save_json(self.output_path, mesh_map)

        missing_links = [label for label in self.link_labels if label not in mesh_map]
        return {
            "output": str(self.output_path),
            "assigned_links": len(mesh_map),
            "assigned_meshes": sum(1 for item in assignments if item != UNASSIGNED),
            "unassigned_meshes": sum(1 for item in assignments if item == UNASSIGNED),
            "links_without_mesh": missing_links,
            "mesh_map": mesh_map,
        }

    def mesh_path(self, file_name: str) -> Optional[Path]:
        part = self.parts_by_file.get(file_name)
        if part is None:
            return None
        path = Path(part["path"]).resolve()
        if not path.exists() or not path.is_file():
            return None
        return path


def _html_page(state: MeshMapWebState) -> bytes:
    title = f"SPARK Mesh Map - {state.metadata_path.name}"
    viewer_script = html.escape(state.viewer_script_url, quote=True)
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script type="module" src="{viewer_script}"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #16202f;
      --muted: #5c6878;
      --accent: #1d6f67;
      --accent-dark: #15564f;
      --warn: #a65f00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 64px;
      padding: 12px 22px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    button {{
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 9px 13px;
      background: var(--accent);
      color: #fff;
      font-weight: 650;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    main {{
      display: grid;
      grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
      gap: 16px;
      padding: 16px 22px 28px;
    }}
    aside {{
      align-self: start;
      position: sticky;
      top: 82px;
      display: grid;
      gap: 12px;
    }}
    section, .mesh-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    section {{
      padding: 14px;
      overflow: hidden;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .path {{
      margin: 6px 0 0;
      color: var(--muted);
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 7px 5px;
      border-bottom: 1px solid #edf0f4;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
    }}
    code {{
      padding: 1px 4px;
      border-radius: 4px;
      background: #edf2f4;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }}
    #mesh-list {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 12px;
      min-width: 0;
    }}
    .mesh-card {{
      display: grid;
      grid-template-rows: auto minmax(260px, 1fr) auto;
      min-height: 392px;
      overflow: hidden;
    }}
    .mesh-title {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      min-width: 0;
    }}
    .mesh-title strong {{
      min-width: 0;
      overflow-wrap: anywhere;
      font-size: 13px;
    }}
    .mesh-title span {{
      color: var(--muted);
      white-space: nowrap;
      font-size: 12px;
    }}
    model-viewer {{
      width: 100%;
      height: 280px;
      background: #f0f3f6;
      --poster-color: #f0f3f6;
    }}
    .mesh-meta {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      padding: 11px 12px 12px;
      border-top: 1px solid var(--line);
    }}
    .kv {{
      color: var(--muted);
      overflow-wrap: anywhere;
      font-size: 12px;
    }}
    select {{
      width: 100%;
      min-height: 36px;
      border: 1px solid #bcc5d1;
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 7px 9px;
      font: inherit;
    }}
    #status {{
      min-height: 20px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    #status.error {{ color: #a12828; }}
    #status.warn {{ color: var(--warn); }}
    pre {{
      max-height: 260px;
      overflow: auto;
      margin: 0;
      padding: 10px;
      border-radius: 6px;
      background: #101820;
      color: #eaf2f1;
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      header {{
        align-items: stretch;
        flex-direction: column;
      }}
      main {{
        grid-template-columns: 1fr;
        padding: 12px;
      }}
      aside {{ position: static; }}
      #mesh-list {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SPARK Mesh Map</h1>
    <button id="save">Save mesh_map.json</button>
  </header>
  <main>
    <aside>
      <section>
        <h2>LLM Link Graph</h2>
        <div id="link-graph"></div>
      </section>
      <section>
        <h2>Paths</h2>
        <div id="paths"></div>
      </section>
      <section>
        <h2>Status</h2>
        <div id="status">Loading</div>
      </section>
      <section>
        <h2>mesh_map.json</h2>
        <pre id="preview">{{}}</pre>
      </section>
    </aside>
    <div id="mesh-list"></div>
  </main>
  <script>
    const state = {{ links: [], parts: [], assignments: {{}}, unassigned: "{UNASSIGNED}" }};
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({{
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }}[ch]));

    function setStatus(text, cls = "") {{
      const el = document.getElementById("status");
      el.textContent = text;
      el.className = cls;
    }}

    function linkOptions(selected) {{
      const values = [state.unassigned, ...state.links.map((link) => link.label)];
      return values.map((label) => {{
        const isSelected = label === selected ? " selected" : "";
        return `<option value="${{escapeHtml(label)}}"${{isSelected}}>${{escapeHtml(label)}}</option>`;
      }}).join("");
    }}

    function renderLinks() {{
      const rows = state.links.map((link) => `
        <tr>
          <td><code>${{escapeHtml(link.label)}}</code></td>
          <td>${{escapeHtml(link.name)}}</td>
          <td><code>${{escapeHtml(link.parent)}}</code></td>
          <td><code>${{escapeHtml(link.joint_type)}}</code></td>
          <td><code>${{escapeHtml(link.axis)}}</code></td>
        </tr>
      `).join("");
      document.getElementById("link-graph").innerHTML = `
        <table>
          <thead><tr><th>Link</th><th>Name</th><th>Parent</th><th>Joint</th><th>Axis</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    function renderPaths() {{
      document.getElementById("paths").innerHTML = `
        <div class="path"><strong>Metadata</strong><br>${{escapeHtml(state.metadata)}}</div>
        <div class="path"><strong>Split Dir</strong><br>${{escapeHtml(state.split_dir)}}</div>
        <div class="path"><strong>Output</strong><br>${{escapeHtml(state.output)}}</div>
      `;
    }}

    function renderMeshes() {{
      document.getElementById("mesh-list").innerHTML = state.parts.map((part) => {{
        const selected = state.assignments[part.file] || state.unassigned;
        return `
          <article class="mesh-card">
            <div class="mesh-title">
              <strong>${{escapeHtml(part.file)}}</strong>
              <span>#${{escapeHtml(part.index)}}</span>
            </div>
            <model-viewer
              src="${{escapeHtml(part.mesh_url)}}"
              camera-controls
              auto-rotate
              shadow-intensity="0.6"
              interaction-prompt="none"
              loading="lazy">
            </model-viewer>
            <div class="mesh-meta">
              <select data-file="${{escapeHtml(part.file)}}">${{linkOptions(selected)}}</select>
              <div class="kv">Geometry: ${{escapeHtml(part.geometry_name)}}<br>Node: ${{escapeHtml(part.node_name)}}<br>Vertices/Faces: ${{escapeHtml(part.vertices)}} / ${{escapeHtml(part.faces)}}</div>
            </div>
          </article>
        `;
      }}).join("");
    }}

    function collectAssignments() {{
      const assignments = {{}};
      for (const select of document.querySelectorAll("select[data-file]")) {{
        assignments[select.dataset.file] = select.value;
      }}
      return assignments;
    }}

    async function loadState() {{
      const response = await fetch("/api/state");
      if (!response.ok) throw new Error(await response.text());
      const loaded = await response.json();
      Object.assign(state, loaded);
      renderLinks();
      renderPaths();
      renderMeshes();
      document.getElementById("preview").textContent = JSON.stringify(state.assignments, null, 2);
      setStatus(`Loaded ${{state.parts.length}} split meshes`);
    }}

    async function saveMapping() {{
      setStatus("Saving");
      const response = await fetch("/api/save", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ assignments: collectAssignments() }})
      }});
      const payload = await response.json().catch(() => ({{ error: "Invalid server response" }}));
      if (!response.ok) {{
        setStatus(payload.error || "Save failed", "error");
        return;
      }}
      document.getElementById("preview").textContent = JSON.stringify(payload.mesh_map, null, 2);
      const missing = payload.links_without_mesh || [];
      if (missing.length) {{
        setStatus(`Saved with missing links: ${{missing.join(", ")}}`, "warn");
      }} else {{
        setStatus(`Saved ${{payload.output}}`);
      }}
    }}

    document.getElementById("save").addEventListener("click", saveMapping);
    loadState().catch((error) => setStatus(error.message, "error"));
  </script>
</body>
</html>"""
    return body.encode("utf-8")


def _make_handler(state: MeshMapWebState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SPARKMeshMapHTTP/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _send_bytes(self, data: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
            self._send_bytes(json.dumps(payload, indent=2).encode("utf-8"), "application/json", status)

        def _send_error_json(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
            self._send_json({"error": message}, status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(_html_page(state), "text/html; charset=utf-8")
                return

            if parsed.path == "/api/state":
                self._send_json(state.to_json())
                return

            if parsed.path.startswith("/mesh/"):
                file_name = unquote(parsed.path.removeprefix("/mesh/"))
                mesh_path = state.mesh_path(file_name)
                if mesh_path is None:
                    self._send_error_json("Mesh file not found", HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(mesh_path.name)[0] or "model/gltf-binary"
                with mesh_path.open("rb") as f:
                    self._send_bytes(f.read(), content_type)
                return

            self._send_error_json("Not found", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/save":
                self._send_error_json("Not found", HTTPStatus.NOT_FOUND)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1024 * 1024:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                assignments = payload.get("assignments", {})
                if not isinstance(assignments, dict):
                    raise ValueError("assignments must be an object")
                self._send_json(state.save_assignments(assignments))
            except Exception as exc:
                self._send_error_json(str(exc), HTTPStatus.BAD_REQUEST)

    return Handler


def serve_app(
    metadata_path: Path,
    split_dir: Path,
    output_path: Path,
    absolute_paths: bool = False,
    host: str = "0.0.0.0",
    port: int = 7860,
    viewer_script_url: str = DEFAULT_VIEWER_SCRIPT,
) -> None:
    state = MeshMapWebState(
        metadata_path=metadata_path,
        split_dir=split_dir,
        output_path=output_path,
        absolute_paths=absolute_paths,
        viewer_script_url=viewer_script_url,
    )
    server = ThreadingHTTPServer((host, port), _make_handler(state))
    display_host = "localhost" if host in ("0.0.0.0", "::") else host
    print(f"Serving SPARK mesh map web app at http://{display_host}:{port}")
    if host == "0.0.0.0":
        print(f"Remote browser URL: http://<server-ip>:{port}")
    print(f"mesh_map.json output: {output_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve split GLB to LLM-link mapping over HTTP.")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--split-dir", required=True, help="Directory containing split GLB files")
    parser.add_argument("--output", required=True, help="Path to write mesh_map.json")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute mesh paths")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--viewer-script-url", default=DEFAULT_VIEWER_SCRIPT)
    args = parser.parse_args()

    serve_app(
        metadata_path=Path(args.metadata),
        split_dir=Path(args.split_dir),
        output_path=Path(args.output),
        absolute_paths=args.absolute_paths,
        host=args.host,
        port=args.port,
        viewer_script_url=args.viewer_script_url,
    )


if __name__ == "__main__":
    main()
