import React, { useEffect, useRef, useState } from "react";
import { apiUrl } from "./lib/api";
import {
  Search,
  Scissors,
  Settings,
  Eye,
  Square,
  Trash2,
  Wand2,
  Save,
  MousePointer2,
  Paintbrush,
  Eraser,
  RotateCcw,
} from "lucide-react";

const CLASS_COLORS = {
  Crack: [255, 0, 0],
  Material_loss: [255, 140, 0],
  Stain: [30, 144, 255],
  "External Fixings": [0, 200, 0],
};

const colorOfClass = (name) => CLASS_COLORS[name] || [255, 255, 0];
const MASK_OVERLAY_ALPHA = 0.52;

const loadImage = (src) =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });

export default function App() {
  const detCanvasRef = useRef(null);
  const segCanvasRef = useRef(null);
  const maskCanvasRef = useRef(document.createElement("canvas"));

  const [level, setLevel] = useState("bbox");
  const [dataset, setDataset] = useState("");
  const [primaryClass, setPrimaryClass] = useState("");
  const [datasets, setDatasets] = useState([]);
  const [index, setIndex] = useState(0);
  const [total, setTotal] = useState(0);
  const [imagePath, setImagePath] = useState("");
  const [imageDataUrl, setImageDataUrl] = useState("");
  const [imageObj, setImageObj] = useState(null);
  const [bboxes, setBboxes] = useState([]);
  const [detPreviewBox, setDetPreviewBox] = useState(null);
  const [detDrawClass, setDetDrawClass] = useState("Crack");
  const [maskClass, setMaskClass] = useState("Crack");
  const [brushSize, setBrushSize] = useState(18);
  const [detMode, setDetMode] = useState("view");
  const [segMode, setSegMode] = useState("view");
  const [detModel, setDetModel] = useState("intersection");
  const [points, setPoints] = useState([]);
  const [brushPreview, setBrushPreview] = useState(null);
  const [detStatus, setDetStatus] = useState("");
  const [segStatus, setSegStatus] = useState("");

  const drawingBoxStart = useRef(null);
  const draggingBoxRef = useRef(null);
  const resizingBoxRef = useRef(null);
  const drawingBrush = useRef(false);

  const fetchJson = async (url, options) => {
    const r = await fetch(url, options);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  };

  const redrawDet = () => {
    if (!imageObj || !detCanvasRef.current) return;
    const c = detCanvasRef.current;
    const ctx = c.getContext("2d");
    c.width = imageObj.width;
    c.height = imageObj.height;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(imageObj, 0, 0);
    bboxes.forEach((b) => {
      const [x, y, w, h] = b.bbox;
      const col = colorOfClass(b.primary_class);
      ctx.strokeStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},0.15)`;
      ctx.fillRect(x, y, w, h);
      // Corner handles for view-mode resize.
      const hs = 6;
      ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
      ctx.fillRect(x - hs / 2, y - hs / 2, hs, hs);
      ctx.fillRect(x + w - hs / 2, y - hs / 2, hs, hs);
      ctx.fillRect(x - hs / 2, y + h - hs / 2, hs, hs);
      ctx.fillRect(x + w - hs / 2, y + h - hs / 2, hs, hs);
    });
    if (detPreviewBox) {
      const [x, y, w, h] = detPreviewBox;
      ctx.strokeStyle = "rgba(250, 204, 21, 0.95)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(250, 204, 21, 0.12)";
      ctx.fillRect(x, y, w, h);
    }
  };

  const redrawSeg = () => {
    if (!imageObj || !segCanvasRef.current) return;
    const c = segCanvasRef.current;
    const ctx = c.getContext("2d");
    c.width = imageObj.width;
    c.height = imageObj.height;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(imageObj, 0, 0);
    const m = maskCanvasRef.current;
    ctx.globalAlpha = MASK_OVERLAY_ALPHA;
    ctx.drawImage(m, 0, 0);
    ctx.globalAlpha = 1;
    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? "#22c55e" : "#ef4444";
      ctx.fill();
    });
    if (
      brushPreview &&
      (segMode === "brush-add" || segMode === "brush-remove")
    ) {
      const [r, g, b] = colorOfClass(maskClass);
      ctx.beginPath();
      ctx.arc(brushPreview.x, brushPreview.y, brushSize / 2, 0, Math.PI * 2);
      ctx.strokeStyle =
        segMode === "brush-remove"
          ? "rgba(239,68,68,0.95)"
          : `rgba(${r},${g},${b},0.95)`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(brushPreview.x, brushPreview.y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle =
        segMode === "brush-remove"
          ? "rgba(239,68,68,0.95)"
          : `rgba(${r},${g},${b},0.95)`;
      ctx.fill();
    }
  };

  const getCanvasPoint = (canvas, e) => {
    const r = canvas.getBoundingClientRect();
    const sx = canvas.width / r.width;
    const sy = canvas.height / r.height;
    return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
  };

  const loadDatasets = async () => {
    const data = await fetchJson(apiUrl(`/api/datasets?level=${encodeURIComponent(level)}`));
    setDatasets(data.datasets || []);
  };

  const loadTotal = async () => {
    const p = new URLSearchParams({ level });
    if (dataset) p.set("dataset", dataset);
    if (primaryClass) p.set("primary_class", primaryClass);
    const data = await fetchJson(apiUrl(`/api/images?${p.toString()}`));
    setTotal(data.total || 0);
    if (index >= (data.total || 0)) setIndex(Math.max(0, (data.total || 1) - 1));
  };

  const loadCurrent = async () => {
    if (!total) {
      setImageObj(null);
      setImageDataUrl("");
      setBboxes([]);
      setPoints([]);
      return;
    }
    const p = new URLSearchParams({ level });
    if (dataset) p.set("dataset", dataset);
    if (primaryClass) p.set("primary_class", primaryClass);
    const data = await fetchJson(apiUrl(`/api/image/${index}?${p.toString()}`));
    if (data.error) throw new Error(data.error);
    setImagePath(data.filepath);
    setImageDataUrl(data.image);
    const img = await loadImage(data.image);
    setImageObj(img);
    setBboxes((data.bboxes || []).map((b, i) => ({ ...b, id: i })));
    const m = maskCanvasRef.current;
    m.width = img.width;
    m.height = img.height;
    const mctx = m.getContext("2d");
    mctx.clearRect(0, 0, m.width, m.height);
    if (data.mask) {
      const mImg = await loadImage(data.mask);
      mctx.drawImage(mImg, 0, 0, m.width, m.height);
    }
    setPoints([]);
  };

  useEffect(() => {
    (async () => {
      try {
        await loadDatasets();
        await loadTotal();
      } catch (e) {
        setDetStatus(String(e.message || e));
      }
    })();
  }, [level]);

  useEffect(() => {
    (async () => {
      try {
        await loadTotal();
      } catch (e) {
        setDetStatus(String(e.message || e));
      }
    })();
  }, [dataset, primaryClass]);

  useEffect(() => {
    (async () => {
      try {
        await loadCurrent();
      } catch (e) {
        setDetStatus(String(e.message || e));
      }
    })();
  }, [index, total, level, dataset, primaryClass]);

  useEffect(() => {
    redrawDet();
    redrawSeg();
  }, [imageObj, bboxes, points, brushPreview, segMode, maskClass, brushSize, detPreviewBox]);

  const runDetection = async () => {
    if (!imageDataUrl) return;
    const data = await fetchJson(apiUrl("/api/detect"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imageDataUrl,
        model: detModel,
        filename: imagePath.split("/").pop() || "unknown.jpg",
      }),
    });
    if (!data.success) throw new Error(data.error || "Detection failed");
    setBboxes((data.bboxes || []).map((b, i) => ({ ...b, id: i })));
    setDetStatus(`Detection finished: ${(data.bboxes || []).length} boxes`);
  };

  const runRefine = async () => {
    if (!imageDataUrl || points.length === 0) return;
    const bboxXyxy = bboxes.map((b) => {
      const [x, y, w, h] = b.bbox;
      return [x, y, x + w, y + h];
    });
    const res = await fetchJson(apiUrl("/api/refine_mask"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imageDataUrl,
        mask_data: maskCanvasRef.current.toDataURL("image/png"),
        points: points.map((p) => [Math.round(p.x), Math.round(p.y)]),
        labels: points.map((p) => p.label),
        bboxes: bboxXyxy,
      }),
    });
    if (!res.success) throw new Error(res.error || "Refinement failed");
    const bin = await loadImage(res.mask);
    const tmp = document.createElement("canvas");
    tmp.width = maskCanvasRef.current.width;
    tmp.height = maskCanvasRef.current.height;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(bin, 0, 0, tmp.width, tmp.height);
    const src = tctx.getImageData(0, 0, tmp.width, tmp.height);
    const dstCtx = maskCanvasRef.current.getContext("2d");
    const dst = dstCtx.getImageData(0, 0, tmp.width, tmp.height);
    const eraseMode = points.every((p) => p.label === 0);
    const [r, g, b] = colorOfClass(maskClass);
    for (let i = 0; i < src.data.length; i += 4) {
      const on = src.data[i] > 10 || src.data[i + 1] > 10 || src.data[i + 2] > 10 || src.data[i + 3] > 10;
      if (!on) continue;
      if (eraseMode) {
        dst.data[i + 3] = 0;
      } else {
        dst.data[i] = r;
        dst.data[i + 1] = g;
        dst.data[i + 2] = b;
        dst.data[i + 3] = 255;
      }
    }
    dstCtx.putImageData(dst, 0, 0);
    setPoints([]);
    redrawSeg();
    setSegStatus("Refinement updated");
  };

  const runAutoSegmentation = async () => {
    if (!imageDataUrl) return;
    const bboxXyxy = bboxes.map((b) => {
      const [x, y, w, h] = b.bbox;
      return [x, y, x + w, y + h];
    });
    const bboxClasses = bboxes.map((b) => b.primary_class || "Material_loss");
    const res = await fetchJson(apiUrl("/api/predict_bboxes"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imageDataUrl,
        bboxes: bboxXyxy,
        bboxes_classes: bboxClasses,
      }),
    });
    if (!res.success) throw new Error(res.error || "Auto segmentation failed");
    const maskImg = await loadImage(res.mask);
    const m = maskCanvasRef.current;
    const mctx = m.getContext("2d");
    mctx.clearRect(0, 0, m.width, m.height);
    mctx.drawImage(maskImg, 0, 0, m.width, m.height);
    setPoints([]);
    redrawSeg();
    setSegStatus("Auto segmentation done (SAM + crack model)");
  };

  const saveAll = async () => {
    const data = await fetchJson(apiUrl("/api/save"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        index,
        level,
        dataset: dataset || null,
        primary_class: primaryClass || null,
        bboxes,
        mask: maskCanvasRef.current.toDataURL("image/png"),
      }),
    });
    if (!data.success) throw new Error(data.error || "Save failed");
    setDetStatus("Saved successfully");
    setSegStatus("Saved successfully");
  };

  const saveChanges = async () => {
    await saveAll();
    setDetStatus("BBox and mask changes saved");
  };

  const handleDetMouseDown = (e) => {
    if (!imageObj) return;
    const p = getCanvasPoint(detCanvasRef.current, e);
    if (detMode === "draw") {
      drawingBoxStart.current = p;
      setDetPreviewBox([p.x, p.y, 0, 0]);
      return;
    }
    if (detMode === "delete") {
      const hit = bboxes.findIndex((b) => {
        const [x, y, w, h] = b.bbox;
        return p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h;
      });
      if (hit >= 0) setBboxes((prev) => prev.filter((_, i) => i !== hit));
      return;
    }
    if (detMode === "view") {
      const handleSize = 8;
      for (let i = bboxes.length - 1; i >= 0; i -= 1) {
        const [x, y, w, h] = bboxes[i].bbox;
        const handles = [
          { name: "tl", x, y },
          { name: "tr", x: x + w, y },
          { name: "bl", x, y: y + h },
          { name: "br", x: x + w, y: y + h },
        ];
        const hitHandle = handles.find(
          (h0) => Math.abs(p.x - h0.x) <= handleSize && Math.abs(p.y - h0.y) <= handleSize
        );
        if (hitHandle) {
          resizingBoxRef.current = {
            index: i,
            handle: hitHandle.name,
            startX: p.x,
            startY: p.y,
            startBox: [...bboxes[i].bbox],
          };
          return;
        }
      }

      for (let i = bboxes.length - 1; i >= 0; i -= 1) {
        const [x, y, w, h] = bboxes[i].bbox;
        if (p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h) {
          draggingBoxRef.current = {
            index: i,
            startX: p.x,
            startY: p.y,
            startBox: [...bboxes[i].bbox],
          };
          return;
        }
      }
    }
  };

  const handleDetMouseMove = (e) => {
    if (!imageObj) return;
    const p1 = getCanvasPoint(detCanvasRef.current, e);
    if (detMode === "draw" && drawingBoxStart.current) {
      const p0 = drawingBoxStart.current;
      const x = Math.min(p0.x, p1.x);
      const y = Math.min(p0.y, p1.y);
      const w = Math.abs(p1.x - p0.x);
      const h = Math.abs(p1.y - p0.y);
      setDetPreviewBox([x, y, w, h]);
      return;
    }
    if (detMode === "view" && draggingBoxRef.current) {
      const { index, startX, startY, startBox } = draggingBoxRef.current;
      const dx = p1.x - startX;
      const dy = p1.y - startY;
      setBboxes((prev) =>
        prev.map((b, i) => (i === index ? { ...b, bbox: [startBox[0] + dx, startBox[1] + dy, startBox[2], startBox[3]] } : b))
      );
      return;
    }
    if (detMode === "view" && resizingBoxRef.current) {
      const { index, handle, startX, startY, startBox } = resizingBoxRef.current;
      const [x0, y0, w0, h0] = startBox;
      const dx = p1.x - startX;
      const dy = p1.y - startY;
      let x = x0;
      let y = y0;
      let w = w0;
      let h = h0;
      if (handle === "tl") {
        x = x0 + dx;
        y = y0 + dy;
        w = w0 - dx;
        h = h0 - dy;
      } else if (handle === "tr") {
        y = y0 + dy;
        w = w0 + dx;
        h = h0 - dy;
      } else if (handle === "bl") {
        x = x0 + dx;
        w = w0 - dx;
        h = h0 + dy;
      } else if (handle === "br") {
        w = w0 + dx;
        h = h0 + dy;
      }
      w = Math.max(4, w);
      h = Math.max(4, h);
      setBboxes((prev) => prev.map((b, i) => (i === index ? { ...b, bbox: [x, y, w, h] } : b)));
    }
  };

  const handleDetMouseUp = (e) => {
    if (!imageObj) return;
    if (detMode === "draw" && drawingBoxStart.current) {
      const p0 = drawingBoxStart.current;
      drawingBoxStart.current = null;
      const p1 = getCanvasPoint(detCanvasRef.current, e);
      const x = Math.min(p0.x, p1.x);
      const y = Math.min(p0.y, p1.y);
      const w = Math.abs(p1.x - p0.x);
      const h = Math.abs(p1.y - p0.y);
      setDetPreviewBox(null);
      if (w < 4 || h < 4) return;
      setBboxes((prev) => [
        ...prev,
        {
          id: Date.now(),
          bbox: [x, y, w, h],
          primary_class: detDrawClass,
          sub_type: detDrawClass,
        },
      ]);
      return;
    }
    draggingBoxRef.current = null;
    resizingBoxRef.current = null;
  };

  const applyBrush = (x, y, erase) => {
    const mctx = maskCanvasRef.current.getContext("2d");
    mctx.save();
    if (erase) {
      mctx.globalCompositeOperation = "source-over";
      mctx.fillStyle = "rgba(0,0,0,1)";
    } else {
      const [r, g, b] = colorOfClass(maskClass);
      mctx.globalCompositeOperation = "source-over";
      mctx.fillStyle = `rgba(${r},${g},${b},1)`;
    }
    mctx.beginPath();
    mctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    mctx.fill();
    mctx.restore();
  };

  const handleSegMouseDown = (e) => {
    if (!imageObj) return;
    const p = getCanvasPoint(segCanvasRef.current, e);
    if (segMode === "point-pos") return setPoints((prev) => [...prev, { x: p.x, y: p.y, label: 1 }]);
    if (segMode === "point-neg") return setPoints((prev) => [...prev, { x: p.x, y: p.y, label: 0 }]);
    if (segMode === "brush-add" || segMode === "brush-remove") {
      drawingBrush.current = true;
      applyBrush(p.x, p.y, segMode === "brush-remove");
      redrawSeg();
    }
  };

  const handleSegMouseMove = (e) => {
    if (!imageObj) return;
    const p = getCanvasPoint(segCanvasRef.current, e);
    setBrushPreview({ x: p.x, y: p.y });
    if (!drawingBrush.current) return;
    applyBrush(p.x, p.y, segMode === "brush-remove");
    redrawSeg();
  };

  const handleSegMouseUp = () => {
    drawingBrush.current = false;
  };

  return (
    <div className="container">
      <div className="card">
        <div className="row">
          <select value={level} onChange={(e) => { setLevel(e.target.value); setIndex(0); }}>
            <option value="bbox">BBox Level</option>
            <option value="patch">Patch Level</option>
          </select>
          <select value={dataset} onChange={(e) => { setDataset(e.target.value); setIndex(0); }}>
            <option value="">All Datasets</option>
            {datasets.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <select value={primaryClass} onChange={(e) => { setPrimaryClass(e.target.value); setIndex(0); }}>
            <option value="">All Classes</option>
            <option value="Crack">Crack</option>
            <option value="Material_loss">Material_loss</option>
            <option value="Stain">Stain</option>
            <option value="External Fixings">External Fixings</option>
          </select>
          <button onClick={() => setIndex((v) => Math.max(0, v - 1))}>Previous</button>
          <button onClick={() => setIndex((v) => Math.min(Math.max(total - 1, 0), v + 1))}>Next</button>
          <button className="primary" onClick={() => saveAll().catch((e) => setDetStatus(e.message))}>Save</button>
          <span>{total ? `Image ${index + 1} / ${total}` : "No images"}</span>
        </div>
        <div className="filePath">Current file: {imagePath || "-"}</div>
      </div>

      <div className="grid">
        <div className="card agentCard">
          <div className="agentHeader">
            <div>
              <h3 className="agentTitle">Detection Agent</h3>
              <p className="agentDesc">Object detection and classification</p>
            </div>
            <button className="ghostBtn iconBtn" onClick={() => setDetStatus("Settings panel ready")}><Settings size={14} /> Settings</button>
          </div>
          <div className="toolbar blockToolbar">
            <label>Model type</label>
            <select value={detModel} onChange={(e) => setDetModel(e.target.value)}>
              <option value="intersection">Intersection</option>
              <option value="ensemble">Ensemble</option>
              <option value="yolo12m">YOLO12m</option>
              <option value="yolo11m">YOLO11m</option>
              <option value="faster-rcnn">Faster R-CNN</option>
              <option value="rtdetr">RT-DETR</option>
              <option value="gpt-4o">gpt-4o</option>
            </select>
            <label>Box Class</label>
            <select value={detDrawClass} onChange={(e) => setDetDrawClass(e.target.value)}>
              <option value="Crack">Crack</option>
              <option value="Material_loss">Material_loss</option>
              <option value="Stain">Stain</option>
              <option value="External Fixings">External Fixings</option>
            </select>
          </div>
          <div className="canvasBox">
            <canvas
              ref={detCanvasRef}
              onMouseDown={handleDetMouseDown}
              onMouseMove={handleDetMouseMove}
              onMouseUp={handleDetMouseUp}
              onMouseLeave={() => {
                drawingBoxStart.current = null;
                draggingBoxRef.current = null;
                resizingBoxRef.current = null;
                setDetPreviewBox(null);
              }}
            />
          </div>
          <div className="toolbar">
            <button className="iconBtn" onClick={() => setDetMode("view")}><Eye size={14} /> View</button>
            <button className="iconBtn" onClick={() => setDetMode("draw")}><Square size={14} /> Draw Box</button>
            <button className="iconBtn" onClick={() => setDetMode("delete")}><Trash2 size={14} /> Delete</button>
            <button className="primary iconBtn" onClick={() => runDetection().catch((e) => setDetStatus(e.message))}><Search size={14} /> Run Detection</button>
            <button className="iconBtn" onClick={() => { setBboxes([]); setDetStatus("All boxes cleared"); }}><RotateCcw size={14} /> Clear all</button>
            <button className="primary iconBtn" onClick={() => saveChanges().catch((e) => setDetStatus(e.message))}><Save size={14} /> Save changes</button>
          </div>
          <div className="legend">
            {Object.keys(CLASS_COLORS).map((k) => {
              const c = CLASS_COLORS[k];
              return <span key={k} className="pill"><span className="dot" style={{ background: `rgb(${c[0]},${c[1]},${c[2]})` }} />{k}</span>;
            })}
          </div>
          <div className="status">{detStatus}</div>
        </div>

        <div className="card agentCard">
          <div className="agentHeader">
            <div>
              <h3 className="agentTitle">Segmentation Agent</h3>
              <p className="agentDesc">Proprietary model segmentation and manual refinement</p>
            </div>
            <button className="ghostBtn iconBtn" onClick={() => setSegStatus("Settings panel ready")}><Settings size={14} /> Settings</button>
          </div>
          <div className="toolbar blockToolbar">
            <div className="rowGroup">
              <label>Mask Class</label>
              <select value={maskClass} onChange={(e) => setMaskClass(e.target.value)}>
                <option value="Crack">Crack</option>
                <option value="Material_loss">Material_loss</option>
                <option value="Stain">Stain</option>
                <option value="External Fixings">External Fixings</option>
              </select>
            </div>
            <div className="rowGroup">
              <label>Brush</label>
              <input type="range" min="2" max="80" value={brushSize} onChange={(e) => setBrushSize(parseInt(e.target.value, 10))} />
              <span className="mutedText">{brushSize}px</span>
            </div>
          </div>
          <div className="canvasBox">
            <canvas
              ref={segCanvasRef}
              onMouseDown={handleSegMouseDown}
              onMouseMove={handleSegMouseMove}
              onMouseUp={handleSegMouseUp}
              onMouseLeave={() => {
                handleSegMouseUp();
                setBrushPreview(null);
              }}
            />
          </div>
          <div className="toolbar">
            <button className="iconBtn" onClick={() => setSegMode("view")}><Eye size={14} /> View</button>
            <button className="iconBtn" onClick={() => setSegMode("point-pos")}><MousePointer2 size={14} /> Positive Point</button>
            <button className="iconBtn" onClick={() => setSegMode("point-neg")}><MousePointer2 size={14} /> Negative Point</button>
            <button className="iconBtn" onClick={() => setSegMode("brush-add")}><Paintbrush size={14} /> Brush</button>
            <button className="iconBtn" onClick={() => setSegMode("brush-remove")}><Eraser size={14} /> Erase</button>
            <button className="primary iconBtn" onClick={() => runAutoSegmentation().catch((e) => setSegStatus(e.message))}><Scissors size={14} /> Run Segmentation</button>
            <button className="primary iconBtn" onClick={() => runRefine().catch((e) => setSegStatus(e.message))}><Wand2 size={14} /> Run Refinement</button>
            <button
              className="iconBtn"
              onClick={() => {
                setPoints([]);
                const m = maskCanvasRef.current;
                const mctx = m.getContext("2d");
                mctx.clearRect(0, 0, m.width, m.height);
                redrawSeg();
                setSegStatus("Mask cleared (save changes to persist)");
              }}
            >
              <RotateCcw size={14} /> Clear
            </button>
            <button className="primary iconBtn" onClick={() => saveChanges().catch((e) => setSegStatus(e.message))}><Save size={14} /> Save changes</button>
          </div>
          <div className="status">{segStatus}</div>
        </div>
      </div>
    </div>
  );
}
