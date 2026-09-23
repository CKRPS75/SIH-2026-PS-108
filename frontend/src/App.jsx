import React, { useState } from 'react';
import { 
  Search, ShieldCheck, Database, Globe, ChevronRight, Activity, 
  UploadCloud, FileText, AlertCircle, CheckCircle, Download, 
  Languages, FileSpreadsheet, ArrowRight, Layers, BookOpen, Scale
} from 'lucide-react';

export default function StandardWiseApp() {
  const [activeView, setActiveView] = useState('landing');

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans selection:bg-blue-200">
      {/* Official Government Top Bar */}
      <div className="bg-[#0f172a] text-white py-1.5 px-6 text-xs font-medium flex justify-between items-center relative z-50">
        <div className="flex items-center space-x-4">
          <span className="tracking-wide text-slate-300"><span className="text-white">GOVERNMENT OF INDIA</span> | MINISTRY OF CONSUMER AFFAIRS</span>
        </div>
        <div className="hidden md:flex space-x-4 text-slate-300">
          <button className="hover:text-white transition-colors">Skip to main content</button>
          <div className="flex items-center space-x-1 border-l border-slate-700 pl-4">
            <button className="hover:text-white">A-</button>
            <button className="hover:text-white">A</button>
            <button className="hover:text-white">A+</button>
          </div>
          <select className="bg-transparent border-none outline-none cursor-pointer hover:text-white border-l border-slate-700 pl-4">
            <option value="en" className="text-black">English</option>
            <option value="hi" className="text-black">हिन्दी</option>
            <option value="mr" className="text-black">मराठी</option>
          </select>
        </div>
      </div>

      {/* Main Navigation */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center space-x-4 cursor-pointer group" onClick={() => setActiveView('landing')}>
            <div className="w-12 h-12 bg-gradient-to-br from-blue-700 to-indigo-900 rounded-xl flex items-center justify-center shadow-lg shadow-blue-900/20 group-hover:scale-105 transition-transform">
              <ShieldCheck size={28} className="text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
                StandardWise <span className="text-orange-500 font-medium text-lg bg-orange-50 px-2 py-0.5 rounded border border-orange-100">मानक-AI</span>
              </h1>
              <p className="text-xs text-blue-700 font-semibold tracking-wide uppercase">Bureau of Indian Standards</p>
            </div>
          </div>
          <nav className="hidden lg:flex space-x-2 bg-slate-100/50 p-1 rounded-lg border border-slate-200">
            <NavBtn active={activeView === 'landing'} onClick={() => setActiveView('landing')}>Home</NavBtn>
            <NavBtn active={activeView === 'dashboard'} onClick={() => setActiveView('dashboard')}>Procurement Console</NavBtn>
            <NavBtn active={activeView === 'batch'} onClick={() => setActiveView('batch')}>BoQ Analysis</NavBtn>
          </nav>
        </div>
      </header>

      {/* Dynamic Content Rendering */}
      <main className="flex-1">
        {activeView === 'landing' && <LandingView onLaunch={() => setActiveView('dashboard')} />}
        {activeView === 'dashboard' && <ProcurementConsole />}
        {activeView === 'batch' && <BoQUploadView />}
      </main>
    </div>
  );
}

function NavBtn({ active, onClick, children }) {
  return (
    <button 
      onClick={onClick}
      className={`px-5 py-2 text-sm font-semibold rounded-md transition-all duration-200 ${
        active 
          ? 'bg-white text-blue-700 shadow-sm border border-slate-200/60' 
          : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'
      }`}
    >
      {children}
    </button>
  );
}

function LandingView({ onLaunch }) {
  return (
    <div className="animate-in fade-in duration-500">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-white border-b border-slate-200">
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-5"></div>
        <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-blue-50 to-transparent"></div>
        
        <div className="max-w-7xl mx-auto px-6 py-20 lg:py-28 relative z-10 flex flex-col lg:flex-row items-center">
          <div className="lg:w-3/5 pr-8">
            <div className="inline-flex items-center space-x-2 bg-blue-50 border border-blue-100 rounded-full px-4 py-1.5 mb-6">
              <span className="flex h-2 w-2 rounded-full bg-blue-600 animate-pulse"></span>
              <span className="text-xs font-bold text-blue-700 tracking-wide uppercase">AI-Powered Recommendation Engine</span>
            </div>
            <h1 className="text-4xl lg:text-6xl font-extrabold text-slate-900 leading-tight mb-6 tracking-tight">
              Procurement specifications, <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-700 to-indigo-600">
                intelligently standardized.
              </span>
            </h1>
            <p className="text-lg text-slate-600 mb-8 max-w-xl leading-relaxed">
              Bridge the semantic gap between colloquial trade terms and formal bureaucratic standard titles. StandardWise cross-references your BoQ against 21,000+ active BIS codes instantly.
            </p>
            <div className="flex flex-wrap gap-4">
              <button 
                onClick={onLaunch}
                className="bg-blue-700 hover:bg-blue-800 text-white px-8 py-3.5 rounded-lg font-semibold transition-all shadow-lg shadow-blue-700/20 flex items-center space-x-2 hover:-translate-y-0.5"
              >
                <span>Launch Search Console</span>
                <ArrowRight size={18} />
              </button>
              <button className="bg-white hover:bg-slate-50 border border-slate-300 text-slate-700 px-8 py-3.5 rounded-lg font-semibold transition-all flex items-center space-x-2 shadow-sm">
                <BookOpen size={18} />
                <span>Read API Docs</span>
              </button>
            </div>
          </div>
          
          <div className="lg:w-2/5 mt-12 lg:mt-0 relative">
            <div className="bg-white p-6 rounded-2xl shadow-2xl border border-slate-200 relative z-10 transform lg:rotate-2 hover:rotate-0 transition-transform duration-500">
               <div className="flex items-center justify-between border-b border-slate-100 pb-4 mb-4">
                  <div className="flex space-x-2">
                     <div className="w-3 h-3 rounded-full bg-red-400"></div>
                     <div className="w-3 h-3 rounded-full bg-amber-400"></div>
                     <div className="w-3 h-3 rounded-full bg-green-400"></div>
                  </div>
                  <span className="text-xs font-mono text-slate-400">/api/v1/search/product-aware</span>
               </div>
               <div className="space-y-4 font-mono text-sm">
                  <div className="bg-slate-50 p-3 rounded border border-slate-100">
                     <span className="text-purple-600 font-semibold">"query"</span>: <span className="text-green-600">"CCTV night vision outdoor"</span>
                  </div>
                  <div className="flex justify-center text-slate-400"><Activity size={16} /></div>
                  <div className="bg-blue-50 p-3 rounded border border-blue-100">
                     <span className="text-blue-700 font-semibold">Found:</span> IS 16910 / IEC 62676<br/>
                     <span className="text-slate-500 text-xs mt-1 block">Video surveillance systems for use in security applications</span>
                  </div>
               </div>
            </div>
            {/* Decorative background blobs */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
            <div className="absolute top-1/2 left-1/4 -translate-x-1/2 -translate-y-1/4 w-72 h-72 bg-indigo-400 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-slate-900 mb-4">Core Platform Capabilities</h2>
          <p className="text-slate-600 max-w-2xl mx-auto">Built on advanced NLP and graph databases to ensure complete compliance under General Financial Rules (GFR) 2017.</p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8">
          <FeatureCard 
            icon={<Database className="text-blue-600" size={24} />}
            title="Hybrid Vector Retrieval"
            description="Combines Dense Vector Search (BGE-M3) with BM25 sparse exact-match filtering for precision recall."
          />
          <FeatureCard 
            icon={<Globe className="text-indigo-600" size={24} />}
            title="Multilingual NLP"
            description="Process freeform text specifications in Hindi, Marathi, and 10+ regional languages via IndicTrans2."
          />
          <FeatureCard 
            icon={<Layers className="text-teal-600" size={24} />}
            title="Dependency Graph"
            description="Traverse Neo4j nodes to auto-discover normative references, test methods, and IP safety codes."
          />
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm hover:shadow-lg transition-all hover:-translate-y-1 group cursor-default">
      <div className="w-14 h-14 rounded-xl bg-slate-50 flex items-center justify-center mb-6 border border-slate-100 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-xl font-bold text-slate-900 mb-3">{title}</h3>
      <p className="text-slate-600 leading-relaxed text-sm">{description}</p>
    </div>
  );
}

function ProcurementConsole() {
  const [product, setProduct] = useState('Portland pozzolana cement');
  const [description, setDescription] = useState('manufactured using fly ash for construction');
  const [language, setLanguage] = useState('en');
  const [limit, setLimit] = useState(5);
  
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const requestJson = { product, description, limit };

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null); setResponse(null);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/search/product-aware', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestJson)
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: Backend Service Unavailable`);
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-8 animate-in fade-in duration-300">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-slate-900">Tender Specification Analyzer</h2>
        <p className="text-slate-600 mt-1">Cross-reference colloquial trade terms against 21,000+ active Indian Standards.</p>
      </div>

      <div className="grid lg:grid-cols-12 gap-8">
        {/* Input Form Column */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
            <div className="bg-slate-50/80 px-6 py-4 border-b border-slate-200 font-semibold text-slate-800 flex items-center space-x-2">
              <FileText size={18} className="text-blue-600" />
              <span>Query Parameters</span>
            </div>
            
            <form onSubmit={handleSearch} className="p-6 space-y-6">
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">Input Language</label>
                <div className="relative">
                  <Languages size={16} className="absolute left-3 top-3 text-slate-400" />
                  <select 
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-full pl-10 pr-4 py-2.5 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-600 focus:border-blue-600 bg-white transition-shadow"
                  >
                    <option value="en">English (Default)</option>
                    <option value="hi">Hindi (IndicTrans2 Enabled)</option>
                    <option value="mr">Marathi (IndicTrans2 Enabled)</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">Product Category <span className="text-red-500">*</span></label>
                <input 
                  type="text" 
                  value={product}
                  onChange={(e) => setProduct(e.target.value)}
                  className="w-full px-4 py-2.5 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-600 focus:border-blue-600 transition-shadow"
                  placeholder="e.g., CCTV surveillance camera"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">Technical Description <span className="text-red-500">*</span></label>
                <textarea 
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full px-4 py-3 border border-slate-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-600 focus:border-blue-600 h-28 resize-none transition-shadow"
                  placeholder="Provide detailed specs, voltage, materials, etc."
                  required
                />
              </div>

              <div className="pt-2 flex space-x-3">
                <button 
                  type="submit" 
                  disabled={loading}
                  className="flex-1 bg-blue-700 hover:bg-blue-800 text-white py-3 rounded-lg font-semibold text-sm transition-all disabled:opacity-70 flex justify-center items-center shadow-md shadow-blue-700/20"
                >
                  {loading ? <Activity className="animate-spin mr-2" size={18} /> : <Search className="mr-2" size={18} />}
                  Retrieve Standards
                </button>
                <button 
                  type="button"
                  onClick={() => { setProduct(''); setDescription(''); }}
                  className="px-5 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 py-3 rounded-lg font-semibold text-sm transition-colors"
                >
                  Clear
                </button>
              </div>
            </form>
          </div>
          
          {/* Debug Request Payload */}
          <div className="bg-[#0f172a] rounded-xl shadow-lg overflow-hidden border border-slate-700/50">
             <div className="px-5 py-3 bg-slate-900/50 text-xs font-mono text-slate-400 border-b border-slate-800 flex justify-between items-center">
                <span>POST /api/v1/search/product-aware</span>
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
             </div>
             <div className="p-5 text-sm font-mono text-emerald-400 overflow-x-auto">
                <pre>{JSON.stringify(requestJson, null, 2)}</pre>
             </div>
          </div>
        </div>

        {/* Results Column */}
        <div className="lg:col-span-7">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-5 flex items-start space-x-4 text-red-800 shadow-sm mb-6">
              <AlertCircle className="shrink-0 mt-0.5 text-red-600" size={24} />
              <div>
                <h4 className="font-bold text-base mb-1">Connection Failed</h4>
                <p className="text-sm text-red-700/90">{error}</p>
                <p className="text-sm mt-2 font-medium">Ensure FastAPI backend and Qdrant DB are running on port 8000.</p>
              </div>
            </div>
          )}

          {!response && !error && !loading && (
             <div className="h-[600px] border-2 border-dashed border-slate-200 rounded-2xl flex flex-col items-center justify-center p-12 text-center bg-white shadow-sm">
                <div className="w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mb-6">
                  <Database size={36} className="text-blue-300" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 mb-2">Awaiting Query Request</h3>
                <p className="text-slate-500 max-w-sm leading-relaxed">
                  Enter procurement details to query the hybrid vector database and traverse the Neo4j normative dependencies.
                </p>
             </div>
          )}

          {response && (
            <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
              
              {/* Compliance & Export Toolbar */}
              <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center bg-white p-5 border border-slate-200 rounded-xl shadow-sm gap-4">
                 <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
                       <CheckCircle className="text-green-600" size={20} />
                    </div>
                    <div>
                       <div className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Tender Health</div>
                       <div className="font-bold text-slate-900 text-lg">98% Compliant</div>
                    </div>
                 </div>
                 <button className="flex items-center justify-center space-x-2 bg-slate-900 hover:bg-slate-800 text-white px-5 py-2.5 rounded-lg text-sm font-semibold transition-colors shadow-md">
                    <Download size={16} />
                    <span>Export GeM Document</span>
                 </button>
              </div>

              {/* Mandatory QCO Alert */}
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 flex items-start space-x-4 shadow-sm relative overflow-hidden">
                 <div className="absolute top-0 left-0 w-1 h-full bg-amber-500"></div>
                 <Scale className="shrink-0 mt-0.5 text-amber-600" size={24} />
                 <div>
                   <h4 className="font-bold text-amber-900 text-base mb-1">Mandatory QCO Certification</h4>
                   <p className="text-sm text-amber-800 leading-relaxed">This product category falls under Compulsory BIS Registration. Final tenders must legally explicitly mandate ISI Mark / Scheme II compliance.</p>
                 </div>
              </div>

              {/* Raw Backend Response */}
              <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden flex flex-col h-[500px]">
                <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex justify-between items-center">
                  <div className="font-bold text-slate-800 flex items-center gap-2">
                     <Database size={16} className="text-slate-400" />
                     Backend Retrieval Results
                  </div>
                  <span className="bg-green-100 text-green-700 text-xs px-2.5 py-1 rounded-md font-mono font-bold border border-green-200">HTTP 200 OK</span>
                </div>
                <div className="p-0 overflow-auto flex-1 bg-[#0f172a]">
                  <pre className="text-sm text-blue-300 font-mono p-6">
                    {JSON.stringify(response, null, 2)}
                  </pre>
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function BoQUploadView() {
  return (
    <div className="w-full max-w-5xl mx-auto px-6 py-12 animate-in fade-in duration-300">
      <div className="text-center mb-10">
        <h2 className="text-3xl font-bold text-slate-900 mb-3">BoQ Batch Analysis</h2>
        <p className="text-slate-600 max-w-2xl mx-auto">Upload complete Tender RFPs or Excel BoQ spreadsheets to automatically extract all line items and generate a comprehensive standards mapping table.</p>
      </div>

      <div className="bg-white border-2 border-dashed border-blue-200 rounded-2xl p-20 flex flex-col items-center justify-center text-center hover:bg-blue-50/30 hover:border-blue-400 transition-colors cursor-pointer shadow-sm">
        <div className="w-20 h-20 bg-blue-100 text-blue-700 rounded-full flex items-center justify-center mb-6 shadow-inner">
          <UploadCloud size={40} />
        </div>
        <h3 className="text-xl font-bold text-slate-800 mb-3">Drag and drop tender documents here</h3>
        <p className="text-slate-500 text-sm mb-8 max-w-md leading-relaxed">
          Supports .PDF (RFP Specifications) and .XLSX / .CSV (Bill of Quantities). Our extraction engine automatically parses text tables and complex clauses.
        </p>
        <button className="bg-white border-2 border-slate-200 text-slate-700 hover:border-slate-300 hover:bg-slate-50 px-8 py-3 rounded-lg font-bold text-sm transition-all flex items-center shadow-sm">
          <FileSpreadsheet size={18} className="mr-2 text-green-600" />
          Browse Files
        </button>
      </div>
    </div>
  );
}