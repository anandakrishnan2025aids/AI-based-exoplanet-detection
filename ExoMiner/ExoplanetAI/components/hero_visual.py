"""
hero_visual.py - Interactive Astronomical Transit Visualizer Component.
Provides a synchronized star-orbit-planet simulation alongside the corresponding
photometric light curve transit dip.
"""

import streamlit as st
import streamlit.components.v1 as components


def render_interactive_transit_hero(height: int = 380):
    """
    Renders the synchronized planetary transit and light curve dip simulator.
    """
    html_code = """
    <!DOCTYPE html>
    <html style="background: #040814; color-scheme: dark;">
    <head>
        <meta charset="utf-8">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            html, body {
                background: #040814 !important;
                color: #E2E8F0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                overflow: hidden;
                padding: 6px;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 8px;
                border: 1px solid rgba(0, 216, 246, 0.25);
                border-radius: 8px;
                padding: 12px;
                background: rgba(8, 15, 30, 0.9);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
            }
            .top-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 11px;
                letter-spacing: 0.08em;
                color: #00D8F6;
                text-transform: uppercase;
                border-bottom: 1px solid rgba(0, 216, 246, 0.15);
                padding-bottom: 6px;
            }
            .canvas-row {
                display: flex;
                gap: 12px;
                height: 250px;
            }
            .orbit-pane {
                flex: 1.1;
                position: relative;
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }
            .lc-pane {
                flex: 1.4;
                position: relative;
            }
            canvas {
                width: 100%;
                height: 100%;
                display: block;
            }
            .controls-row {
                display: flex;
                align-items: center;
                gap: 12px;
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
            }
            button {
                background: rgba(0, 216, 246, 0.2);
                border: 1px solid #00D8F6;
                color: #FFFFFF;
                border-radius: 4px;
                padding: 4px 10px;
                font-family: inherit;
                cursor: pointer;
                transition: all 0.2s;
            }
            button:hover {
                background: rgba(0, 216, 246, 0.4);
            }
            input[type=range] {
                flex: 1;
                accent-color: #00D8F6;
                cursor: pointer;
            }
            .readout {
                color: #00D8F6;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="top-bar">
                <span>Astronomical Transit Photometry Simulator</span>
                <span>ΔF / F ≈ (R_p / R_*)²</span>
                <span id="transit-tag" style="color: #64748B;">Out of Transit</span>
            </div>

            <div class="canvas-row">
                <div class="orbit-pane">
                    <canvas id="orbitCanvas"></canvas>
                </div>
                <div class="lc-pane">
                    <canvas id="lcCanvas"></canvas>
                </div>
            </div>

            <div class="controls-row">
                <button id="playBtn">Pause</button>
                <span>Phase:</span>
                <input type="range" id="phaseSlider" min="0" max="1" step="0.002" value="0.5">
                <span>Flux: <span id="fluxReadout" class="readout">1.0000</span></span>
                <span style="margin-left: auto; color: #94A3B8;">Drag slider to inspect transit dip</span>
            </div>
        </div>

        <script>
            const orbitC = document.getElementById('orbitCanvas');
            const lcC = document.getElementById('lcCanvas');
            const oCtx = orbitC.getContext('2d');
            const lCtx = lcC.getContext('2d');
            const playBtn = document.getElementById('playBtn');
            const slider = document.getElementById('phaseSlider');
            const fluxEl = document.getElementById('fluxReadout');
            const tagEl = document.getElementById('transit-tag');

            let isPlaying = true;
            let phase = 0.5;
            const transitCenter = 0.5;
            const transitWidth = 0.08;
            const transitDepth = 0.035; // 3.5% dip

            function resize() {
                orbitC.width = orbitC.clientWidth * window.devicePixelRatio;
                orbitC.height = orbitC.clientHeight * window.devicePixelRatio;
                lcC.width = lcC.clientWidth * window.devicePixelRatio;
                lcC.height = lcC.clientHeight * window.devicePixelRatio;
            }
            window.addEventListener('resize', resize);
            resize();

            playBtn.addEventListener('click', () => {
                isPlaying = !isPlaying;
                playBtn.textContent = isPlaying ? 'Pause' : 'Play';
            });

            slider.addEventListener('input', (e) => {
                isPlaying = false;
                playBtn.textContent = 'Play';
                phase = parseFloat(e.target.value);
            });

            function getFlux(p) {
                const diff = Math.abs(p - transitCenter);
                if (diff < transitWidth) {
                    const factor = Math.cos((diff / transitWidth) * (Math.PI / 2));
                    return 1.0 - transitDepth * Math.pow(factor, 2);
                }
                return 1.0;
            }

            function draw() {
                const oW = orbitC.width;
                const oH = orbitC.height;
                const lW = lcC.width;
                const lH = lcC.height;

                oCtx.clearRect(0, 0, oW, oH);
                lCtx.clearRect(0, 0, lW, lH);

                // 1. Draw Star and Orbit
                const cx = oW * 0.5;
                const cy = oH * 0.5;
                const rx = oW * 0.42;
                const ry = oH * 0.18;
                const starR = oH * 0.22;

                // Star Glow
                const glow = oCtx.createRadialGradient(cx, cy, starR * 0.2, cx, cy, starR * 1.5);
                glow.addColorStop(0, 'rgba(254, 240, 138, 1)');
                glow.addColorStop(0.3, 'rgba(245, 158, 11, 0.8)');
                glow.addColorStop(0.7, 'rgba(0, 216, 246, 0.2)');
                glow.addColorStop(1, 'transparent');
                oCtx.fillStyle = glow;
                oCtx.beginPath();
                oCtx.arc(cx, cy, starR * 1.5, 0, Math.PI * 2);
                oCtx.fill();

                // Orbit path
                oCtx.strokeStyle = 'rgba(0, 216, 246, 0.25)';
                oCtx.setLineDash([4, 4]);
                oCtx.beginPath();
                oCtx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
                oCtx.stroke();
                oCtx.setLineDash([]);

                // Star Body
                oCtx.fillStyle = '#FEF08A';
                oCtx.beginPath();
                oCtx.arc(cx, cy, starR, 0, Math.PI * 2);
                oCtx.fill();

                // Planet position
                const angle = (phase * Math.PI * 2) - (Math.PI / 2);
                const px = cx + rx * Math.cos(angle);
                const py = cy + ry * Math.sin(angle);
                const isFront = py > cy;
                const planetR = starR * 0.28;

                // Planet
                oCtx.fillStyle = isFront ? '#030712' : '#64748B';
                oCtx.strokeStyle = '#00D8F6';
                oCtx.lineWidth = 2;
                oCtx.beginPath();
                oCtx.arc(px, py, planetR, 0, Math.PI * 2);
                oCtx.fill();
                oCtx.stroke();

                // Orbit label
                oCtx.fillStyle = '#94A3B8';
                oCtx.font = `${Math.floor(oH * 0.055)}px monospace`;
                oCtx.fillText("Host Star", cx - 35, cy - starR - 10);

                // 2. Draw Synchronized Light Curve
                const padL = lW * 0.12;
                const padR = lW * 0.05;
                const padT = lH * 0.18;
                const padB = lH * 0.22;
                const plotW = lW - padL - padR;
                const plotH = lH - padT - padB;

                // Grid lines & labels
                lCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
                lCtx.lineWidth = 1;
                lCtx.strokeRect(padL, padT, plotW, plotH);

                lCtx.fillStyle = '#64748B';
                lCtx.font = `${Math.floor(lH * 0.05)}px monospace`;
                lCtx.fillText("1.000", padL - 38, padT + 10);
                lCtx.fillText("0.965", padL - 38, padT + plotH);
                lCtx.fillText("Phase: 0.0", padL, padT + plotH + 18);
                lCtx.fillText("Phase: 0.5 (Transit)", padL + plotW * 0.35, padT + plotH + 18);
                lCtx.fillText("Phase: 1.0", padL + plotW - 65, padT + plotH + 18);

                // Baseline reference
                lCtx.strokeStyle = 'rgba(0, 216, 246, 0.2)';
                lCtx.setLineDash([3, 3]);
                lCtx.beginPath();
                lCtx.moveTo(padL, padT + 10);
                lCtx.lineTo(padL + plotW, padT + 10);
                lCtx.stroke();
                lCtx.setLineDash([]);

                // Full light curve curve
                lCtx.strokeStyle = '#00D8F6';
                lCtx.lineWidth = 2.5;
                lCtx.beginPath();
                for (let i = 0; i <= 200; i++) {
                    const p = i / 200;
                    const f = getFlux(p);
                    const x = padL + p * plotW;
                    const y = padT + 10 + (1.0 - f) / transitDepth * (plotH - 18);
                    if (i === 0) lCtx.moveTo(x, y);
                    else lCtx.lineTo(x, y);
                }
                lCtx.stroke();

                // Live cadence marker on light curve
                const curFlux = getFlux(phase);
                const curX = padL + phase * plotW;
                const curY = padT + 10 + (1.0 - curFlux) / transitDepth * (plotH - 18);

                lCtx.fillStyle = '#FFFFFF';
                lCtx.strokeStyle = '#00D8F6';
                lCtx.lineWidth = 2;
                lCtx.beginPath();
                lCtx.arc(curX, curY, 5, 0, Math.PI * 2);
                lCtx.fill();
                lCtx.stroke();

                // Vertical indicator
                lCtx.strokeStyle = 'rgba(0, 216, 246, 0.5)';
                lCtx.setLineDash([2, 2]);
                lCtx.beginPath();
                lCtx.moveTo(curX, padT);
                lCtx.lineTo(curX, padT + plotH);
                lCtx.stroke();
                lCtx.setLineDash([]);

                // Update UI text
                fluxEl.textContent = curFlux.toFixed(4);
                if (curFlux < 0.999 && isFront) {
                    tagEl.textContent = "TRANSIT IN PROGRESS (Dip: -" + ((1.0 - curFlux) * 100).toFixed(2) + "%)";
                    tagEl.style.color = "#00D8F6";
                } else {
                    tagEl.textContent = "Out of Transit";
                    tagEl.style.color = "#64748B";
                }

                // Increment phase if playing
                if (isPlaying) {
                    phase = (phase + 0.003) % 1.0;
                    slider.value = phase.toFixed(3);
                }

                requestAnimationFrame(draw);
            }

            draw();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)
