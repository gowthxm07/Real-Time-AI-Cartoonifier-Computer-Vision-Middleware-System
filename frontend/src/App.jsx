import React, { useState, useEffect } from 'react';
import { Camera, Upload, ArrowLeft, Cpu, Sparkles, Activity, ShieldAlert, Layers, Radio, Zap, Download, Target, CheckCircle, TrendingDown } from 'lucide-react';
import './App.css';

function App() {
  const [mode, setMode] = useState('home');
  const [uploadStatus, setUploadStatus] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSnapping, setIsSnapping] = useState(false);

  // BENCHMARK STATES
  const [benchTime, setBenchTime] = useState(15);
  const [benchResults, setBenchResults] = useState(null);

  // ENTERPRISE ROI SLIDER STATES
  const [cameraCount, setCameraCount] = useState(100);
  const [operatingHours, setOperatingHours] = useState(24);

  // CONTROL STATES
  const [noiseIntensity, setNoiseIntensity] = useState(50);
  const [autoMode, setAutoMode] = useState(false);
  const [motionAware, setMotionAware] = useState(false);
  const [privacyMode, setPrivacyMode] = useState(false);
  const [iotCompression, setIotCompression] = useState(false);
  const [pipelineMode, setPipelineMode] = useState('standard');
  const [selectedChannel, setSelectedChannel] = useState('cartoon');

  const [metrics, setMetrics] = useState({
    raw_entropy: 0.0, proc_entropy: 0.0, entropy_reduction: 0.0, variance_drop: 0.0, compute_savings: 0.0, identity_risk: 'NONE', fps: 0.0, latency: 0.0, cpu: 0.0
  });

  const channels = [
    { id: 'cartoon', label: 'Final Output' }, { id: 'edges', label: 'Edge Detect' }, { id: 'color', label: 'Color Quantized' }, { id: 'silhouette', label: 'Silhouette' }, { id: 'mask', label: 'Binary Mask' }
  ];
  const pipelineModes = [
    { id: 'standard', label: 'Standard Mode' }, { id: 'tracking', label: 'Tracking Optimized' }, { id: 'privacy', label: 'Privacy Mode' }, { id: 'ar', label: 'AR Overlay Mode' }, { id: 'edge', label: 'Edge Compute Mode' }
  ];

  useEffect(() => {
    let interval;
    if (isPlaying && mode === 'studio') {
      interval = setInterval(async () => {
        try {
          const res = await fetch('http://localhost:5000/metrics');
          if (res.ok) setMetrics(await res.json());
        } catch (e) { }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, mode]);

  const updateProcessingConfig = async (intensity, auto, motion, privacy, iot, pipeline) => {
    await fetch('http://localhost:5000/set_processing_config', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ intensity, auto_mode: auto, motion_aware: motion, privacy_mode: privacy, iot_compression: iot, pipeline_mode: pipeline })
    });
  };

  const handlePipelineChange = (selectedMode) => {
    setPipelineMode(selectedMode);
    let newIntensity = noiseIntensity, newAuto = autoMode, newMotion = motionAware, newPrivacy = privacyMode, newIot = iotCompression;
    switch(selectedMode) {
      case 'tracking': newIntensity = 80; newAuto = false; newMotion = true; newPrivacy = false; newIot = false; break;
      case 'privacy': newIntensity = 60; newAuto = false; newMotion = false; newPrivacy = true; newIot = false; break;
      case 'ar': newIntensity = 20; newAuto = true; newMotion = false; newPrivacy = false; newIot = false; break;
      case 'edge': newIntensity = 10; newAuto = false; newMotion = false; newPrivacy = false; newIot = true; break;
      case 'standard': default: newIntensity = 50; newAuto = false; newMotion = false; newPrivacy = false; newIot = false; break;
    }
    setNoiseIntensity(newIntensity); setAutoMode(newAuto); setMotionAware(newMotion); setPrivacyMode(newPrivacy); setIotCompression(newIot);
    updateProcessingConfig(newIntensity, newAuto, newMotion, newPrivacy, newIot, selectedMode);
  };

  const handleIntensityChange = async (value) => {
    setNoiseIntensity(value); setPipelineMode('custom');
    await updateProcessingConfig(value, autoMode, motionAware, privacyMode, iotCompression, 'custom');
  };

  const handleToggle = async (type, value) => {
    setPipelineMode('custom');
    let a=autoMode, m=motionAware, p=privacyMode, i=iotCompression;
    if(type==='auto') { a=value; setAutoMode(value); }
    if(type==='motion') { m=value; setMotionAware(value); }
    if(type==='privacy') { p=value; setPrivacyMode(value); }
    if(type==='iot') { i=value; setIotCompression(value); }
    await updateProcessingConfig(noiseIntensity, a, m, p, i, 'custom');
  };

  const handleSnapshot = async () => {
    if (!isPlaying) return;
    setIsSnapping(true);
    try {
      const res = await fetch('http://localhost:5000/snapshot');
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `pipeline_snapshot_${Date.now()}.zip`;
        document.body.appendChild(a); a.click(); a.remove();
      }
    } catch (e) { }
    setIsSnapping(false);
  };

  const runLiveBenchmark = async () => {
    if (!isPlaying) return;
    setMode('benchmark');
    setBenchResults(null);
    setBenchTime(15);
    
    await fetch('http://localhost:5000/start_benchmark', { method: 'POST' });
    
    const poll = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:5000/benchmark_status');
        const data = await res.json();
        setBenchTime(data.time_left);
        
        if (!data.is_running && data.stats) {
            clearInterval(poll);
            setBenchResults(data.stats);
        }
      } catch(e) {}
    }, 500);
  };

  const activateWebcam = async () => {
    setMode('studio'); setIsPlaying(false); setSelectedChannel('cartoon');
    await fetch('http://localhost:5000/set_mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ type: 'webcam' }) });
    setTimeout(() => setIsPlaying(true), 700);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]; if (!file) return;
    setMode('studio'); setUploadStatus('Uploading...'); setIsPlaying(false); setSelectedChannel('cartoon');
    const formData = new FormData(); formData.append('file', file);
    const res = await fetch('http://localhost:5000/upload', { method: 'POST', body: formData });
    if (res.ok) { setUploadStatus('Processing...'); setTimeout(() => setIsPlaying(true), 700); }
  };

  const backHome = async () => {
    setIsPlaying(false); setMode('home');
    await fetch('http://localhost:5000/set_mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ type: 'stop' }) });
  };

  // --- ROI CALCULATOR MATH ---
  const calculateROI = () => {
    if (!benchResults) return { rawCost: 0, procCost: 0, savings: 0 };
    
    const fps = 30;
    const awsCostPerGB = 0.023; // Standard AWS S3 pricing per GB
    
    // Calculate GB per year per camera
    const rawGbPerYear = (benchResults.raw_kb * fps * 3600 * operatingHours * 365) / (1024 * 1024);
    const procGbPerYear = (benchResults.proc_kb * fps * 3600 * operatingHours * 365) / (1024 * 1024);
    
    const rawCost = rawGbPerYear * cameraCount * awsCostPerGB;
    const procCost = procGbPerYear * cameraCount * awsCostPerGB;
    
    return {
      rawCost: rawCost,
      procCost: procCost,
      savings: rawCost - procCost
    };
  };

  const formatMoney = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="app">
      <header className="navbar">
        <div className="logo"><Sparkles size={20} /><span>Cartoonify Studio</span></div>
        <div className="chip"><Cpu size={14} /> Optimized Runtime</div>
      </header>

      {mode === 'home' && (
        <main className="home">
          <div className="hero-glow"></div>
          <div className="home-content">
            <div className="version-badge">v2.0 Enterprise Pipeline</div>
            <h1>Real-Time AI <span>Cartoonifier</span></h1>
            <p>Transform live or recorded video streams into simplified outputs using adaptive computer vision. Built for Edge deployment, IoT compression, and Privacy compliance.</p>
            <div className="home-actions">
              <button className="primary-btn" onClick={activateWebcam}><Camera size={20} />Start Live Feed</button>
              <label className="secondary-btn"><Upload size={20} />Upload Video<input type="file" hidden accept="video/*" onChange={handleFileUpload} /></label>
            </div>
            <div className="features-grid">
              <div className="feature-card"><div className="feature-icon"><Activity size={24} /></div><h3>Motion Tracking</h3><p>MOG2 background subtraction suppresses static environments to focus downstream AI on moving subjects.</p></div>
              <div className="feature-card"><div className="feature-icon"><ShieldAlert size={24} /></div><h3>Privacy Anonymization</h3><p>Real-time Haar Cascade facial blurring guarantees strict GDPR & HIPAA visual compliance.</p></div>
              <div className="feature-card"><div className="feature-icon"><Radio size={24} /></div><h3>IoT Compression</h3><p>Extreme 6-bit spatial macro-blocking saves up to 95% bandwidth for weak network transmission.</p></div>
              <div className="feature-card"><div className="feature-icon"><Cpu size={24} /></div><h3>Edge Compute</h3><p>Dynamic 240p downscaling ensures smooth, high-FPS inference on severely limited IoT hardware.</p></div>
            </div>
          </div>
        </main>
      )}

      {mode === 'studio' && (
        <main className="studio">
          <div className="studio-header">
            <button onClick={backHome} className="back"><ArrowLeft size={18} />Back</button>
            <div className="header-actions">
              <button className="benchmark-btn" onClick={runLiveBenchmark} disabled={!isPlaying}>
                <Target size={14} /> Run AI Benchmark
              </button>
              <div className="status">{isPlaying ? 'Live Processing' : uploadStatus || 'Initializing'}</div>
            </div>
          </div>

          <div className="control-panel">
            <div className="pipeline-switcher">
              <span className="switcher-label"><Layers size={16}/> Pipeline Mode:</span>
              <div className="switcher-buttons">
                {pipelineModes.map(m => (
                  <button key={m.id} className={`pipeline-btn ${pipelineMode === m.id ? 'active' : ''}`} onClick={() => handlePipelineChange(m.id)}>{m.label}</button>
                ))}
              </div>
            </div>
            <div className="control-group" style={{ marginTop: '20px' }}>
              <label>Noise Reduction Intensity: <strong>{noiseIntensity}</strong></label>
              <input type="range" min="0" max="100" value={noiseIntensity} disabled={autoMode} onChange={(e) => handleIntensityChange(e.target.value)} />
            </div>
            <div className="toggle-group">
              <label className="checkbox-label"><input type="checkbox" checked={autoMode} onChange={(e) => handleToggle('auto', e.target.checked)} />Auto Adaptive Mode</label>
              <label className="checkbox-label highlight-checkbox"><input type="checkbox" checked={motionAware} onChange={(e) => handleToggle('motion', e.target.checked)} />Motion-Aware Simplification</label>
              <label className="checkbox-label privacy-checkbox"><input type="checkbox" checked={privacyMode} onChange={(e) => handleToggle('privacy', e.target.checked)} /><ShieldAlert size={14} /> Privacy Anonymization</label>
              <label className="checkbox-label iot-checkbox"><input type="checkbox" checked={iotCompression} onChange={(e) => handleToggle('iot', e.target.checked)} /><Radio size={14} /> IoT 6-Bit Compression</label>
            </div>
          </div>

          {isPlaying && (
            <div className="metrics-panel">
              <div className="metric-box"><span className="metric-label">Raw Entropy</span><span className="metric-value">{Number(metrics.raw_entropy).toFixed(2)} <small>bits</small></span></div>
              <div className="metric-box"><span className="metric-label">Output Entropy</span><span className="metric-value">{Number(metrics.proc_entropy).toFixed(2)} <small>bits</small></span></div>
              <div className="metric-box highlight-metric"><span className="metric-label">Data Reduction</span><span className="metric-value">{Number(metrics.entropy_reduction).toFixed(1)}%</span></div>
              <div className="metric-box highlight-metric"><span className="metric-label">Variance Drop</span><span className="metric-value">{Number(metrics.variance_drop).toFixed(1)}%</span></div>
              <div className="metric-box success-metric"><span className="metric-label"><Activity size={14}/> Compute Savings</span><span className="metric-value">{Number(metrics.compute_savings).toFixed(1)}%</span></div>
              <div className={`metric-box ${metrics.identity_risk === 'HIGH' ? 'danger-metric' : 'safe-metric'}`}><span className="metric-label"><ShieldAlert size={14}/> Identity Risk</span><span className="metric-value">{metrics.identity_risk}</span></div>
              <div className="metric-box hardware-metric"><span className="metric-label"><Zap size={14}/> Est. FPS</span><span className="metric-value">{Number(metrics.fps).toFixed(1)}</span></div>
              <div className="metric-box hardware-metric"><span className="metric-label">Latency</span><span className="metric-value">{Number(metrics.latency).toFixed(0)} <small>ms</small></span></div>
              <div className="metric-box hardware-metric"><span className="metric-label">CPU Load</span><span className="metric-value">{Number(metrics.cpu).toFixed(1)}%</span></div>
            </div>
          )}

          <div className="video-grid">
            <div className="video-card">
              <div className="card-header-flex">
                <div className="card-title-row"><div className="card-title">Source Feed</div></div>
                <div className="channel-tabs" style={{ visibility: 'hidden', pointerEvents: 'none' }}><button className="channel-tab">Spacer</button></div>
              </div>
              <div className="video-box">{isPlaying ? <img src="http://localhost:5000/video_feed/raw" alt="Raw" /> : <div className="loader">Starting stream...</div>}</div>
            </div>

            <div className="video-card highlight">
              <div className="card-header-flex">
                <div className="card-title-row">
                  <div className="card-title">Analysis API Channel</div>
                  <button className="snapshot-btn" onClick={handleSnapshot} disabled={!isPlaying || isSnapping}>
                    <Download size={14} /> {isSnapping ? 'Saving...' : 'Export Snapshot'}
                  </button>
                </div>
                <div className="channel-tabs">
                  {channels.map(ch => (
                    <button key={ch.id} className={`channel-tab ${selectedChannel === ch.id ? 'active' : ''}`} onClick={() => setSelectedChannel(ch.id)}>{ch.label}</button>
                  ))}
                </div>
              </div>
              <div className="video-box">{isPlaying ? <img src={`http://localhost:5000/video_feed/${selectedChannel}`} alt="Feed" /> : <div className="loader">Processing...</div>}</div>
            </div>
          </div>
        </main>
      )}

      {/* --- NEW BENCHMARK LIVE EVALUATION ROOM --- */}
      {mode === 'benchmark' && (
        <main className="benchmark-room">
          {!benchResults ? (
            <div className="benchmark-active">
               <h2 className="pulse-text">YOLOv8 AI Currently Evaluating...</h2>
               <div className="countdown-timer">{benchTime}s</div>
               <p className="warning-text">Expect frame drops: Running dual neural networks simultaneously.</p>
               
               <div className="video-grid benchmark-video-grid">
                  <div className="video-card danger-card">
                    <div className="card-title"><Target size={16}/> YOLOv8 on RAW Feed</div>
                    <div className="video-box"><img src="http://localhost:5000/benchmark_feed/raw" alt="YOLO Raw" /></div>
                  </div>
                  <div className="video-card safe-card">
                    <div className="card-title"><Target size={16}/> YOLOv8 on PROCESSED Feed</div>
                    <div className="video-box"><img src="http://localhost:5000/benchmark_feed/processed" alt="YOLO Processed" /></div>
                  </div>
               </div>
            </div>
          ) : (
            <div className="benchmark-report" style={{ maxWidth: '1000px' }}>
               <div className="report-header">
                 <CheckCircle size={40} color="#00ffa3" />
                 <h2>Enterprise ROI Report</h2>
                 <p>15-Second Automated Live Benchmark Complete</p>
               </div>
               
               {/* 4-BOX GRID */}
               <div className="report-grid">
                  <div className="report-box">
                    <h3>Network Data (JPEG Transmission)</h3>
                    <div className="stat-row"><span>Raw Data Sent:</span> <span>{benchResults.raw_kb} KB/frame</span></div>
                    <div className="stat-row"><span>Processed Data Sent:</span> <span>{benchResults.proc_kb} KB/frame</span></div>
                    <div className="report-highlight">{benchResults.savings_percent}% Bandwidth Saved</div>
                  </div>
                  
                  <div className="report-box">
                    <h3>AI Cognitive Load (False Positives)</h3>
                    <div className="stat-row"><span>Objects Tracked in Raw:</span> <span>{benchResults.raw_objs} per frame</span></div>
                    <div className="stat-row"><span>Objects Tracked in Processed:</span> <span>{benchResults.proc_objs} per frame</span></div>
                    <div className="report-highlight">{benchResults.objs_prevented} Irrelevant Objects Ignored</div>
                  </div>

                  <div className="report-box">
                    <h3>Computational Complexity (Contours)</h3>
                    <div className="stat-row"><span>Raw Video Shapes:</span> <span>{benchResults.raw_contours} edges</span></div>
                    <div className="stat-row"><span>Processed Shapes:</span> <span>{benchResults.proc_contours} edges</span></div>
                    <div className="report-highlight">{benchResults.contours_prevented} Useless Textures Deleted</div>
                  </div>

                  <div className="report-box">
                    <h3>Algorithmic Processing Speed</h3>
                    <div className="stat-row"><span>Raw Edge Detection Time:</span> <span>{benchResults.raw_cv_time} ms</span></div>
                    <div className="stat-row"><span>Processed Detection Time:</span> <span>{benchResults.proc_cv_time} ms</span></div>
                    <div className="report-highlight">{benchResults.time_saved_percent}% Faster Execution Time</div>
                  </div>
               </div>

               {/* --- NEW ENTERPRISE ROI SLIDER --- */}
               <div className="roi-calculator">
                 <div className="roi-header">
                   <TrendingDown size={24} color="#4da3ff"/>
                   <h3>Projected Annual AWS Storage Savings</h3>
                 </div>
                 
                 <div className="roi-controls">
                   <div className="roi-slider-group">
                     <label>Fleet Size: <strong>{cameraCount} Cameras</strong></label>
                     <input type="range" min="1" max="1000" step="10" value={cameraCount} onChange={(e) => setCameraCount(Number(e.target.value))} />
                   </div>
                   <div className="roi-slider-group">
                     <label>Active Time: <strong>{operatingHours} Hours/Day</strong></label>
                     <input type="range" min="1" max="24" step="1" value={operatingHours} onChange={(e) => setOperatingHours(Number(e.target.value))} />
                   </div>
                 </div>

                 <div className="roi-results">
                   <div className="roi-stat"><span>Raw AWS Cost:</span> <span>{formatMoney(calculateROI().rawCost)} / yr</span></div>
                   <div className="roi-stat"><span>Processed AWS Cost:</span> <span>{formatMoney(calculateROI().procCost)} / yr</span></div>
                   <div className="roi-massive-saving">
                     {formatMoney(calculateROI().savings)} Saved Annually
                   </div>
                   <p className="roi-footnote">*Calculation based on live benchmark bandwidth multiplied by standard AWS S3 storage pricing ($0.023/GB) at 30 FPS.</p>
                 </div>
               </div>

               <button className="primary-btn mt-20" style={{ margin: '30px auto 0' }} onClick={() => setMode('studio')}>Return to Studio</button>
            </div>
          )}
        </main>
      )}
    </div>
  );
}

export default App;