import os
import io
import time
import base64
import threading
import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_from_directory

from models.spnet_base import SPNet
from models.aspnet_plus import ASPNetPlus

app = Flask(__name__)
# Enable up to 32MB uploads if needed
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models on {device}...")

inference_lock = threading.Lock()

# 1. Load Baseline SPNet
base_model = SPNet().to(device)
if os.path.exists(r"weights\spnet_baseline.pth"):
    base_model.load_state_dict(torch.load(r"weights\spnet_baseline.pth", map_location=device))
    base_model.eval()
    print("Baseline SPNet loaded successfully.")

# 2. Load Proposed A-SPNet+
upg_model = ASPNetPlus().to(device)
if os.path.exists(r"weights\aspnet_plus.pth"):
    upg_model.load_state_dict(torch.load(r"weights\aspnet_plus.pth", map_location=device))
    upg_model.eval()
    print("Upgraded A-SPNet+ loaded successfully.")

def channelwise_he(img_np):
    """Channel-wise RGB Histogram Equalization preserving chromaticity."""
    img_u8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    res = np.zeros_like(img_u8)
    for c in range(3):
        res[:, :, c] = cv2.equalizeHist(img_u8[:, :, c])
    return res.astype(np.float32) / 255.0

def restore_vivid_color(raw_input_np, enhanced_lum_np, saturation_scale=1.15):
    """
    Retinex Color Coupling (Guo et al. LIME & Retinex Theory):
    Couples the network's enhanced structural luminance with the authentic 
    chromaticity spectrum of the input image, restoring vibrant, natural color.
    """
    lum_in = 0.299 * raw_input_np[:, :, 0] + 0.587 * raw_input_np[:, :, 1] + 0.114 * raw_input_np[:, :, 2] + 1e-4
    lum_enh = 0.299 * enhanced_lum_np[:, :, 0] + 0.587 * enhanced_lum_np[:, :, 1] + 0.114 * enhanced_lum_np[:, :, 2]

    # Ratio of illumination enhancement
    ratio = np.clip(lum_enh / lum_in, 0.0, 20.0)[:, :, np.newaxis]

    # Project chromaticity
    color_projected = np.clip(raw_input_np * ratio, 0.0, 1.0)

    # Enhance color vibrance in HSV space
    hsv = cv2.cvtColor((color_projected * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    vivid_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return vivid_rgb

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Low-Light Image Enhancement Studio | RGMCET ECE</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.3);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #10b981;
            --error: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); min-height: 100vh; padding: 2rem 1rem; }
        .container { max-width: 1240px; margin: 0 auto; }
        
        header { text-align: center; margin-bottom: 2rem; }
        .badge { display: inline-block; padding: 0.35rem 0.85rem; border-radius: 9999px; background: rgba(56,189,248,0.15); color: var(--accent); font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem; border: 1px solid var(--accent); }
        h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .meta { color: var(--text-muted); font-size: 0.95rem; }
        
        .grid-panel { display: grid; grid-template-columns: 360px 1fr; gap: 1.5rem; }
        @media(max-width: 860px) { .grid-panel { grid-template-columns: 1fr; } }
        
        .card { background: var(--card-bg); border-radius: 1rem; padding: 1.5rem; border: 1px solid var(--border); box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3); }
        .card h2 { font-size: 1.15rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
        
        .upload-zone { 
            border: 2px dashed var(--border); 
            border-radius: 0.75rem; 
            padding: 1.8rem 1rem; 
            text-align: center; 
            cursor: pointer; 
            transition: all 0.25s ease; 
            background: rgba(15,23,42,0.5); 
        }
        .upload-zone:hover, .upload-zone.dragover { 
            border-color: var(--accent); 
            background: rgba(56,189,248,0.08); 
            transform: scale(1.01);
        }
        .upload-icon { font-size: 2.4rem; color: var(--accent); margin-bottom: 0.4rem; }
        
        .sample-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
        .chip { background: #334155; padding: 0.45rem 0.8rem; border-radius: 0.5rem; font-size: 0.82rem; cursor: pointer; border: 1px solid transparent; transition: 0.2s; font-weight: 500; }
        .chip:hover { border-color: var(--accent); background: #475569; }
        .chip.active { background: #0284c7; border-color: var(--accent); color: white; box-shadow: 0 0 10px rgba(56,189,248,0.4); }
        .chip.highlight { border-color: #f59e0b; color: #fbbf24; }
        
        .radio-group { display: flex; flex-direction: column; gap: 0.75rem; margin: 1.25rem 0; }
        .radio-item { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem; background: #0f172a; border-radius: 0.5rem; border: 1px solid var(--border); cursor: pointer; transition: 0.2s; }
        .radio-item:hover { border-color: var(--accent); }
        .radio-item input { accent-color: var(--accent); width: 1.15rem; height: 1.15rem; }
        
        .btn { width: 100%; padding: 0.9rem; border: none; border-radius: 0.6rem; background: linear-gradient(135deg, #0284c7, #6366f1); color: white; font-weight: 600; font-size: 1rem; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 15px var(--accent-glow); display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
        .btn:hover { transform: translateY(-1px); filter: brightness(1.1); }
        .btn:disabled { opacity: 0.65; cursor: not-allowed; }

        .btn-download { background: #334155; font-size: 0.85rem; padding: 0.5rem 1rem; border-radius: 0.4rem; color: #fff; text-decoration: none; border: 1px solid var(--border); transition: 0.2s; display: inline-flex; align-items: center; gap: 0.4rem; }
        .btn-download:hover { background: #475569; border-color: var(--accent); }
        
        .alert-box { padding: 0.75rem 1rem; border-radius: 0.5rem; margin-top: 1rem; font-size: 0.85rem; display: none; }
        .alert-box.error { background: rgba(239, 68, 68, 0.15); border: 1px solid var(--error); color: #fca5a5; display: block; }
        .alert-box.success { background: rgba(16, 185, 129, 0.15); border: 1px solid var(--success); color: #86efac; display: block; }
        
        .viewer-container { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; position: relative; }
        @media(max-width: 600px) { .viewer-container { grid-template-columns: 1fr; } }
        
        .img-box { background: #0f172a; border-radius: 0.75rem; padding: 0.75rem; border: 1px solid var(--border); text-align: center; position: relative; min-height: 380px; display: flex; flex-direction: column; }
        .img-box-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
        .img-box h3 { font-size: 0.95rem; color: var(--text-muted); }
        .img-box img { width: 100%; height: 350px; object-fit: contain; border-radius: 0.5rem; background: #000; flex: 1; }
        
        .spinner-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(15,23,42,0.85); display: none; flex-direction: column; align-items: center; justify-content: center; border-radius: 0.75rem; z-index: 10; backdrop-filter: blur(2px); }
        .spinner-overlay.show { display: flex; }
        .spinner { width: 44px; height: 44px; border: 4px solid rgba(56,189,248,0.2); border-top: 4px solid var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; margin-bottom: 0.75rem; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .stats-bar { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.25rem; }
        .stat-item { background: #0f172a; border-radius: 0.5rem; padding: 0.75rem; text-align: center; border: 1px solid var(--border); }
        .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; }
        .stat-val { font-size: 1.25rem; font-weight: 700; color: var(--accent); margin-top: 0.2rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <span class="badge">Final Year B.Tech Project • Batch 60</span>
            <h1>Adaptive Low-Light Image Enhancement</h1>
            <p class="meta">Department of ECE, RGMCET (Autonomous) • K. Pradeep (23091A04D9) & K. Pavani (23091A04D7)</p>
        </header>

        <div class="grid-panel">
            <!-- Sidebar Controls -->
            <div class="card">
                <h2>⚙️ Control Panel</h2>
                
                <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📸</div>
                    <strong>Click or Drag Image Here</strong>
                    <p style="font-size:0.8rem; color:var(--text-muted); margin-top:0.3rem;">Supports JPG, PNG, WEBP, BMP</p>
                    <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="handleFile(this.files[0])">
                </div>

                <div style="margin-top: 1.25rem;">
                    <span style="font-size:0.85rem; color:var(--text-muted);">Quick Test Benchmarks & Custom Photos:</span>
                    <div class="sample-chips">
                        <span class="chip highlight active" onclick="loadSample('custom_ceiling.jpg', this)">🏮 Ceiling Lamps (Your Image)</span>
                        <span class="chip" onclick="loadSample('179.png', this)">Tassels (Fig. 5)</span>
                        <span class="chip" onclick="loadSample('22.png', this)">Totoro (Fig. 5)</span>
                        <span class="chip" onclick="loadSample('780.png', this)">Blue Floor (Fig. 5)</span>
                        <span class="chip" onclick="loadSample('1.png', this)">Bookshelf</span>
                        <span class="chip" onclick="loadSample('146.png', this)">Wardrobe</span>
                    </div>
                </div>

                <div class="radio-group">
                    <label class="radio-item">
                        <input type="radio" name="modelType" value="upgraded" checked>
                        <div>
                            <strong>A-SPNet+ (Our Upgraded Model)</strong>
                            <p style="font-size:0.75rem; color:var(--text-muted);">Wavelet Denoising + SALEM Exposure (0.56M)</p>
                        </div>
                    </label>
                    <label class="radio-item">
                        <input type="radio" name="modelType" value="baseline">
                        <div>
                            <strong>SPNet (Base Paper Model)</strong>
                            <p style="font-size:0.75rem; color:var(--text-muted);">IEEE TMM 2025 Architecture (0.85M)</p>
                        </div>
                    </label>
                </div>

                <button class="btn" id="enhanceBtn" onclick="enhanceImage()">⚡ Enhance Image Live</button>
                <div id="alertBox" class="alert-box"></div>
            </div>

            <!-- Viewer & Results -->
            <div class="card">
                <h2>✨ Real-Time Enhancement Viewer</h2>
                <div class="viewer-container">
                    <div class="img-box">
                        <div class="img-box-header">
                            <h3>Original Low-Light Input</h3>
                        </div>
                        <img id="inputPreview" src="/sample/custom_ceiling.jpg" alt="Input Preview">
                    </div>
                    
                    <div class="img-box">
                        <div class="img-box-header">
                            <h3>Vivid Enhanced Output</h3>
                            <a id="downloadBtn" href="#" download="enhanced_output.png" class="btn-download" style="display:none;">⬇️ Save</a>
                        </div>
                        <img id="outputPreview" src="/sample/custom_ceiling.jpg" alt="Enhanced Result">
                        
                        <div class="spinner-overlay" id="spinnerOverlay">
                            <div class="spinner"></div>
                            <strong style="font-size:0.95rem; color:#fff;">Enhancing with Deep Neural Network...</strong>
                            <p style="font-size:0.75rem; color:var(--text-muted); margin-top:0.25rem;">Performing Retinex Decomposition & Wavelet Denoising</p>
                        </div>
                    </div>
                </div>

                <div class="stats-bar">
                    <div class="stat-item">
                        <div class="stat-label">Inference Latency</div>
                        <div class="stat-val" id="latencyVal">-- ms</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Model Parameters</div>
                        <div class="stat-val" id="paramsVal">0.56 M</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Brightness Boost</div>
                        <div class="stat-val" id="gainVal">-- %</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentImageBase64 = null;
        let isEnhancing = false;

        // Setup Drag & Drop Handlers
        const dropZone = document.getElementById('dropZone');
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            }, false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            }, false);
        });
        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        function showAlert(msg, isError) {
            const box = document.getElementById('alertBox');
            box.className = 'alert-box ' + (isError ? 'error' : 'success');
            box.innerText = msg;
            box.style.display = 'block';
            setTimeout(() => { box.style.display = 'none'; }, 6000);
        }

        function loadSample(name, chipElem) {
            if (chipElem) {
                document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
                chipElem.classList.add('active');
            }
            fetch('/sample/' + name)
                .then(res => {
                    if (!res.ok) throw new Error("Could not load sample file");
                    return res.blob();
                })
                .then(blob => {
                    handleFile(blob);
                })
                .catch(err => {
                    showAlert("Failed to load sample: " + err.message, true);
                });
        }

        // Downscale large images client-side before sending to server to prevent lag and OOM
        function resizeImageClientSide(file, maxDimension, callback) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = new Image();
                img.onload = function() {
                    let w = img.width;
                    let h = img.height;
                    if (w > maxDimension || h > maxDimension) {
                        if (w > h) {
                            h = Math.round((h * maxDimension) / w);
                            w = maxDimension;
                        } else {
                            w = Math.round((w * maxDimension) / h);
                            h = maxDimension;
                        }
                    }
                    const canvas = document.createElement('canvas');
                    canvas.width = w;
                    canvas.height = h;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, w, h);
                    const resizedB64 = canvas.toDataURL('image/jpeg', 0.95);
                    callback(resizedB64);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        function handleFile(file) {
            if (!file) return;
            resizeImageClientSide(file, 1200, function(b64) {
                currentImageBase64 = b64;
                document.getElementById('inputPreview').src = b64;
                enhanceImage();
            });
        }

        window.onload = function() {
            loadSample('custom_ceiling.jpg', document.querySelector('.chip.active'));
        };

        function enhanceImage() {
            if (!currentImageBase64 || isEnhancing) return;
            isEnhancing = true;

            const modelType = document.querySelector('input[name="modelType"]:checked').value;
            const btn = document.getElementById('enhanceBtn');
            const spinner = document.getElementById('spinnerOverlay');
            const downloadBtn = document.getElementById('downloadBtn');

            btn.innerText = "⏳ Enhancing...";
            btn.disabled = true;
            spinner.classList.add('show');

            fetch('/enhance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: currentImageBase64, model: modelType })
            })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(j => { throw new Error(j.error || "Server returned status " + res.status); });
                }
                return res.json();
            })
            .then(data => {
                btn.innerText = "⚡ Enhance Image Live";
                btn.disabled = false;
                spinner.classList.remove('show');
                isEnhancing = false;

                if (data.error) {
                    showAlert("Enhancement error: " + data.error, true);
                    return;
                }

                document.getElementById('outputPreview').src = data.enhanced;
                document.getElementById('latencyVal').innerText = data.latency + " ms";
                document.getElementById('paramsVal').innerText = data.params;
                document.getElementById('gainVal').innerText = "+" + data.boost + "%";

                downloadBtn.href = data.enhanced;
                downloadBtn.style.display = 'inline-flex';
                showAlert("Enhanced successfully! Brightness boost: +" + data.boost + "%", false);
            })
            .catch(err => {
                btn.innerText = "⚡ Enhance Image Live";
                btn.disabled = false;
                spinner.classList.remove('show');
                isEnhancing = false;
                showAlert("Failed to enhance: " + err.message, true);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/sample/<filename>')
def get_sample(filename):
    sample_dir = os.path.abspath(r"data\test\LOL\low")
    return send_from_directory(sample_dir, filename)

@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided in request'}), 400

        img_b64 = data['image'].split(',')[-1]
        model_choice = data.get('model', 'upgraded')

        img_bytes = base64.b64decode(img_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Limit maximum dimension to 1200 to prevent OOM on raw camera photos
        max_dim = 1200
        w, h = pil_img.size
        if max(w, h) > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int((h * max_dim) / w)
            else:
                new_h = max_dim
                new_w = int((w * max_dim) / h)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = pil_img.size

        # Ensure dimensions are multiples of 4 for clean wavelet/conv operations
        crop_w = w - (w % 4)
        crop_h = h - (h % 4)
        if crop_w != w or crop_h != h:
            pil_img = pil_img.crop((0, 0, crop_w, crop_h))

        img_np = np.array(pil_img, dtype=np.float32) / 255.0

        # RGB Channel-wise HE
        she_np = channelwise_he(img_np)
        s2_np = np.clip(img_np + 0.5 * img_np * (1.0 - img_np), 0.0, 1.0)

        t_s1 = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).float().to(device)
        t_she = torch.from_numpy(she_np.transpose(2,0,1)).unsqueeze(0).float().to(device)
        t_s2 = torch.from_numpy(s2_np.transpose(2,0,1)).unsqueeze(0).float().to(device)

        with inference_lock:
            t0 = time.time()
            with torch.no_grad():
                if model_choice == 'baseline':
                    _, _, _, _, enh = base_model(t_s1, t_she, t_s2, lam=0.85)
                    params_str = "0.85 M"
                else:
                    _, _, _, _, enh = upg_model(t_s1, t_she, t_s2, global_lam=0.5)
                    params_str = "0.56 M"
            latency = (time.time() - t0) * 1000.0

        enh_np = np.clip(enh.squeeze(0).permute(1,2,0).cpu().numpy(), 0.0, 1.0)

        # Retinex Color Coupling: restores full natural RGB color vibrance
        vivid_output = restore_vivid_color(img_np, enh_np, saturation_scale=1.15)

        # Calculate brightness boost percentage
        in_mean = np.mean(img_np)
        out_mean = np.mean(vivid_output)
        boost = int(((out_mean - in_mean) / (in_mean + 1e-4)) * 100.0)

        enh_uint8 = (vivid_output * 255.0).astype(np.uint8)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(enh_uint8, cv2.COLOR_RGB2BGR))
        enh_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'enhanced': enh_b64,
            'latency': f"{latency:.1f}",
            'params': params_str,
            'boost': max(0, boost)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Vivid-Color Flask Web Studio on http://127.0.0.1:5000 ...")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
