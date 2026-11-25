import {useMemo, useState} from 'react';
import {
  Check,
  Copy,
  Download,
  FileCode2,
  Folder,
  Search,
  Terminal,
} from 'lucide-react';
import {PROJECT_FILES, ProjectFile} from './projectFiles';

type SectionTab = 'codebase' | 'architecture' | 'rag-agent' | 'setup';

interface ModuleLayer {
  name: string;
  folder: string;
  responsibility: string;
  keyApis: string;
}

const ARCHITECTURE_LAYERS: ModuleLayer[] = [
  {
    name: '01. Presentation & Session Orchestration',
    folder: 'app.py',
    responsibility:
      'Renders the ChatGPT-inspired Streamlit workspace with right-aligned user bubbles, left-aligned assistant Markdown/code responses, a 10-section sidebar, fixed bottom input, and real-time st.write_stream() rendering.',
    keyApis: 'st.chat_input, st.chat_message, st.write_stream, st.session_state',
  },
  {
    name: '02. Configuration & Credential Guard',
    folder: 'config/settings.py',
    responsibility:
      'Loads GROQ_API_KEY via python-dotenv, validates key presence without exposing secrets in UI, centralizes model catalog, chunking sizes, and creates local storage folders.',
    keyApis: 'python-dotenv, dataclasses, AppSettings',
  },
  {
    name: '03. Modular Groq LLM Provider',
    folder: 'llm/groq_model.py',
    responsibility:
      'Instantiates ChatGroq with configurable model, temperature, and streaming flags, while mapping raw authentication, rate-limit, and token errors into clear user-facing messages.',
    keyApis: 'langchain_groq.ChatGroq, classify_groq_error',
  },
  {
    name: '04. Conversational Memory & Persistence',
    folder: 'memory/chat_history.py',
    responsibility:
      'Isolates conversation context per UUID session using InMemoryChatMessageHistory hydrated from local JSON files under data/chat_sessions/ so chat histories never mix across sessions.',
    keyApis: 'InMemoryChatMessageHistory, RunnableWithMessageHistory',
  },
  {
    name: '05. Retrieval-Augmented Generation (RAG)',
    folder: 'rag/*',
    responsibility:
      'Validates and loads PDF, TXT, DOCX, and CSV uploads, splits text into overlapping chunks, computes HuggingFace embeddings, persists a local FAISS index, and retrieves top-k citations with page numbers.',
    keyApis: 'PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, FAISS',
  },
  {
    name: '06. LangChain Tool-Calling Agent',
    folder: 'agents/*',
    responsibility:
      'Routes user prompts across Document Search, Safe AST Calculator, Knowledge Retrieval, and General LLM tools using modern create_tool_calling_agent + AgentExecutor with error recovery.',
    keyApis: 'create_tool_calling_agent, AgentExecutor, @tool',
  },
];

const RAG_STEPS = [
  {
    step: '01. File Validation & Storage',
    detail:
      'rag/document_loader.py checks extension (.pdf, .txt, .docx, .csv), non-empty payload, and 25 MB limit before writing sanitized files into data/uploads/.',
  },
  {
    step: '02. Format-Specific Text Extraction',
    detail:
      'Routes uploads to PyPDFLoader, TextLoader (with encoding auto-detection), Docx2txtLoader, or CSVLoader, normalizing source_file and 1-based page_number metadata.',
  },
  {
    step: '03. Semantic Chunking',
    detail:
      'rag/text_splitter.py uses RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200) to preserve paragraph and sentence boundaries while retaining document metadata.',
  },
  {
    step: '04. Local Embeddings & FAISS Indexing',
    detail:
      'rag/embeddings.py caches sentence-transformers/all-MiniLM-L6-v2 on CPU. rag/vector_store.py stores vectors in data/vectorstore/ and maintains manifest.json for clean removal and re-indexing.',
  },
  {
    step: '05. Grounded Retrieval & Citations',
    detail:
      'rag/retriever.py executes similarity_search_with_score and passes formatted excerpts plus strict anti-hallucination instructions so the LLM states when an answer is missing from uploaded files.',
  },
];

const AGENT_TOOLS = [
  {
    name: 'document_search_tool',
    purpose: 'Queries the local FAISS vector index when RAG is enabled and returns excerpts with filenames and page numbers.',
    safety: 'Honors the sidebar RAG ON/OFF toggle and instructs the model not to fabricate missing facts.',
  },
  {
    name: 'calculator_tool',
    purpose: 'Evaluates arithmetic, exponents, and math functions (sqrt, sin, cos, log, round).',
    safety: 'Uses a custom Python ast.parse evaluator rather than eval(), blocking arbitrary code execution.',
  },
  {
    name: 'knowledge_retrieval_tool',
    purpose: 'Looks up concise architectural reference entries on RAG, Groq, LangChain, Embeddings, FAISS, and Streamlit.',
    safety: 'Read-only deterministic lookup with graceful fallback suggestions.',
  },
  {
    name: 'general_llm_tool',
    purpose: 'Handles general knowledge questions and conversational turns that do not require uploaded files or math.',
    safety: 'Uses ChatGroq with normalized error classification via classify_groq_error().',
  },
];

const TROUBLESHOOTING_ITEMS = [
  {
    issue: 'GROQ_API_KEY missing or 401 authentication error',
    resolution:
      'Copy .env.example to .env inside genai-chatbot/, paste a valid gsk_... key from console.groq.com/keys, and restart Streamlit.',
  },
  {
    issue: 'First document upload takes 15-30 seconds',
    resolution:
      'HuggingFace downloads sentence-transformers/all-MiniLM-L6-v2 (~90 MB) on initial run and caches it locally for all subsequent uploads.',
  },
  {
    issue: 'PDF upload returns "No readable text was found"',
    resolution:
      'PyPDFLoader extracts embedded text layers. Scanned image-only PDFs require OCR pre-processing before uploading.',
  },
  {
    issue: '429 Rate Limit error from Groq API',
    resolution:
      'Wait a few seconds before retrying or switch to llama-3.1-8b-instant in the sidebar model selector.',
  },
];

function buildBootstrapScript(files: ProjectFile[]): string {
  const lines: string[] = [
    '#!/usr/bin/env bash',
    'set -euo pipefail',
    '',
    'echo "Creating genai-chatbot project structure..."',
    'mkdir -p genai-chatbot/config genai-chatbot/llm genai-chatbot/agents genai-chatbot/rag genai-chatbot/memory genai-chatbot/utils genai-chatbot/data/uploads genai-chatbot/data/vectorstore genai-chatbot/data/chat_sessions genai-chatbot/assets',
    'touch genai-chatbot/data/uploads/.gitkeep genai-chatbot/data/vectorstore/.gitkeep genai-chatbot/data/chat_sessions/.gitkeep',
    '',
  ];

  for (const file of files) {
    const marker = `EOF_${file.relPath.replace(/[^A-Za-z0-9]/g, '_').toUpperCase()}`;
    lines.push(`cat <<'${marker}' > "${file.path}"`);
    lines.push(file.content.replace(/\n$/, ''));
    lines.push(marker);
    lines.push('');
  }

  lines.push('echo "Project files created in ./genai-chatbot"');
  lines.push('echo "Next steps:"');
  lines.push('echo "  cd genai-chatbot && python -m venv .venv && source .venv/bin/activate"');
  lines.push('echo "  pip install -r requirements.txt && cp .env.example .env"');
  lines.push('echo "  streamlit run app.py"');
  return lines.join('\n');
}

export default function App() {
  const [activeTab, setActiveTab] = useState<SectionTab>('codebase');
  const [selectedPath, setSelectedPath] = useState<string>(PROJECT_FILES[0].path);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const filteredFiles = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return PROJECT_FILES;
    return PROJECT_FILES.filter(
      (f) =>
        f.relPath.toLowerCase().includes(q) ||
        f.description.toLowerCase().includes(q)
    );
  }, [searchQuery]);

  const activeFile = useMemo(
    () => PROJECT_FILES.find((f) => f.path === selectedPath) ?? PROJECT_FILES[0],
    [selectedPath]
  );

  const totalLines = useMemo(
    () =>
      PROJECT_FILES.reduce(
        (acc, file) => acc + file.content.split('\n').length,
        0
      ),
    []
  );

  const handleCopy = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId((curr) => (curr === id ? null : curr)), 1800);
    } catch {
      // Clipboard API unavailable in restricted context
    }
  };

  const handleDownloadBundle = () => {
    const script = buildBootstrapScript(PROJECT_FILES);
    const blob = new Blob([script], {type: 'text/x-shellscript'});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'setup_genai_chatbot.sh';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const handleDownloadSingleFile = (file: ProjectFile) => {
    const blob = new Blob([file.content], {type: 'text/plain;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = file.relPath.split('/').pop() || 'file.txt';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const activeFileLines = activeFile.content.split('\n');

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Top Bar Contract: 3 zones separated by gap-8 */}
      <header className="sticky top-0 z-20 bg-white border-b border-slate-200 px-6 py-3.5">
        <div className="max-w-[1440px] mx-auto flex items-center justify-between gap-8">
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('codebase');
            }}
            className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
          >
            NexusQnA
          </a>

          <nav
            aria-label="Primary sections"
            className="flex items-center gap-6 overflow-x-auto"
          >
            {(
              [
                ['codebase', 'Codebase'],
                ['architecture', 'Architecture'],
                ['rag-agent', 'RAG & Agent'],
                ['setup', 'Setup Guide'],
              ] as Array<[SectionTab, string]>
            ).map(([tabId, label]) => {
              const isCurrent = activeTab === tabId;
              return (
                <button
                  key={tabId}
                  type="button"
                  onClick={() => setActiveTab(tabId)}
                  className={`text-sm font-medium pb-1 border-b-2 transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-600 rounded-sm ${
                    isCurrent
                      ? 'text-slate-900 border-emerald-600'
                      : 'text-slate-500 border-transparent hover:text-slate-900'
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </nav>

          <div className="shrink-0">
            <button
              type="button"
              onClick={handleDownloadBundle}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-600 focus-visible:ring-offset-2"
            >
              <Download className="w-3.5 h-3.5" aria-hidden="true" />
              Download Bundle
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-6">
        {/* Compact Context Banner */}
        <section className="mb-6 flex flex-col md:flex-row md:items-end md:justify-between gap-4 border-b border-slate-200 pb-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900">
              AI-Powered Conversational QnA Chatbot
            </h1>
            <p className="mt-1 text-sm text-slate-600 max-w-3xl">
              Production Python and Streamlit implementation powered by Groq ChatGroq,
              LangChain tool-calling agents, RunnableWithMessageHistory, HuggingFace
              embeddings, and local FAISS retrieval.
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500 tabular-nums shrink-0">
            <span>{PROJECT_FILES.length} source files</span>
            <span aria-hidden="true">·</span>
            <span>{totalLines.toLocaleString()} lines</span>
            <span aria-hidden="true">·</span>
            <span>Python 3.11+ / Streamlit</span>
          </div>
        </section>

        {activeTab === 'codebase' && (
          <div className="grid grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)] gap-6 items-start">
            {/* Sidebar File Tree */}
            <aside className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="relative mb-3">
                <Search
                  className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                  aria-hidden="true"
                />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Filter files..."
                  aria-label="Filter project files"
                  className="w-full pl-9 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:bg-white"
                />
              </div>

              <div className="text-xs font-semibold text-slate-500 mb-2 px-1 flex items-center gap-1.5">
                <Folder className="w-3.5 h-3.5" aria-hidden="true" />
                <span>genai-chatbot</span>
              </div>

              {filteredFiles.length === 0 ? (
                <div className="py-8 text-center">
                  <p className="text-xs text-slate-500 mb-2">No matching files.</p>
                  <button
                    type="button"
                    onClick={() => setSearchQuery('')}
                    className="text-xs font-medium text-emerald-700 hover:underline"
                  >
                    Clear filter
                  </button>
                </div>
              ) : (
                <ul className="space-y-0.5 max-h-[640px] overflow-y-auto pr-1">
                  {filteredFiles.map((file) => {
                    const isSelected = file.path === activeFile.path;
                    return (
                      <li key={file.path}>
                        <button
                          type="button"
                          onClick={() => setSelectedPath(file.path)}
                          className={`w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-left text-xs font-mono transition-colors ${
                            isSelected
                              ? 'bg-emerald-50 text-emerald-900 font-medium'
                              : 'text-slate-700 hover:bg-slate-100'
                          }`}
                        >
                          <FileCode2
                            className={`w-3.5 h-3.5 shrink-0 ${
                              isSelected ? 'text-emerald-600' : 'text-slate-400'
                            }`}
                            aria-hidden="true"
                          />
                          <span className="truncate">{file.relPath}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </aside>

            {/* Code Viewer */}
            <section className="bg-white border border-slate-200 rounded-xl overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-sm font-mono font-medium text-slate-900 truncate">
                    <span>{activeFile.path}</span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    {activeFile.description} ·{' '}
                    <span className="tabular-nums">
                      {activeFileLines.length} lines
                    </span>
                  </p>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    type="button"
                    onClick={() => handleDownloadSingleFile(activeFile)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0"
                  >
                    <Download className="w-3.5 h-3.5" aria-hidden="true" />
                    Save File
                  </button>
                  <button
                    type="button"
                    onClick={() => handleCopy(activeFile.content, activeFile.path)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors whitespace-nowrap shrink-0"
                  >
                    {copiedId === activeFile.path ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-400" aria-hidden="true" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" aria-hidden="true" />
                        Copy Code
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="bg-slate-950 text-slate-100 overflow-x-auto max-h-[680px]">
                <table className="w-full border-collapse text-xs font-mono leading-relaxed">
                  <tbody>
                    {activeFileLines.map((line, index) => (
                      <tr key={index} className="hover:bg-slate-900/70">
                        <td className="select-none text-right pr-4 pl-4 py-0.5 text-slate-500 w-12 align-top tabular-nums">
                          {index + 1}
                        </td>
                        <td className="pr-5 py-0.5 whitespace-pre text-slate-100">
                          {line || ' '}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'architecture' && (
          <div className="space-y-6">
            <section className="bg-white border border-slate-200 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-slate-900">
                Modular System Architecture
              </h2>
              <p className="mt-1 text-sm text-slate-600 max-w-3xl">
                Strict separation of concerns isolates Streamlit presentation from Groq LLM
                credentials, FAISS vector indexing, per-session LangChain memory, and tool
                orchestration.
              </p>

              <div className="mt-5 overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-xs text-slate-500">
                      <th className="py-2.5 pr-4 font-semibold">Layer</th>
                      <th className="py-2.5 pr-4 font-semibold">Module</th>
                      <th className="py-2.5 pr-4 font-semibold">Responsibility</th>
                      <th className="py-2.5 font-semibold">Core APIs</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {ARCHITECTURE_LAYERS.map((layer) => (
                      <tr key={layer.name} className="hover:bg-slate-50/80">
                        <td className="py-3.5 pr-4 font-medium text-slate-900 whitespace-nowrap">
                          {layer.name}
                        </td>
                        <td className="py-3.5 pr-4 font-mono text-xs text-emerald-700 whitespace-nowrap">
                          {layer.folder}
                        </td>
                        <td className="py-3.5 pr-4 text-slate-600 min-w-[280px]">
                          {layer.responsibility}
                        </td>
                        <td className="py-3.5 font-mono text-xs text-slate-500">
                          {layer.keyApis}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="bg-white border border-slate-200 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-slate-900">
                Request Execution Flow
              </h2>
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <div className="font-semibold text-slate-900">1. Session & Input Guard</div>
                  <p className="mt-1.5 text-slate-600 text-xs leading-relaxed">
                    st.chat_input submits text to validate_user_input(). User messages are
                    appended to the active session UUID JSON file and rendered on the right.
                  </p>
                </div>
                <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <div className="font-semibold text-slate-900">2. Mode Routing</div>
                  <p className="mt-1.5 text-slate-600 text-xs leading-relaxed">
                    Agent ON invokes create_tool_calling_agent with 4 tools. Agent OFF runs
                    Direct LCEL streaming with optional FAISS retrieval and RunnableWithMessageHistory.
                  </p>
                </div>
                <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <div className="font-semibold text-slate-900">3. Streaming & Citations</div>
                  <p className="mt-1.5 text-slate-600 text-xs leading-relaxed">
                    Tokens stream left-aligned via st.write_stream(). Invoked tool names and
                    retrieved document citations with page numbers render below the response.
                  </p>
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'rag-agent' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* RAG Pipeline */}
            <section className="bg-white border border-slate-200 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-slate-900">
                Retrieval-Augmented Generation Pipeline
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Optional local document indexing with HuggingFace Embeddings and FAISS
              </p>

              <ol className="mt-5 space-y-4">
                {RAG_STEPS.map((item) => (
                  <li key={item.step} className="pb-4 border-b border-slate-100 last:border-b-0 last:pb-0">
                    <div className="text-sm font-semibold text-slate-900">{item.step}</div>
                    <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                      {item.detail}
                    </p>
                  </li>
                ))}
              </ol>
            </section>

            {/* AI Agent Tools */}
            <section className="bg-white border border-slate-200 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-slate-900">
                LangChain AI Agent Tool Routing
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Modern create_tool_calling_agent + AgentExecutor with isolated session history
              </p>

              <div className="mt-5 space-y-4">
                {AGENT_TOOLS.map((toolItem) => (
                  <div
                    key={toolItem.name}
                    className="pb-4 border-b border-slate-100 last:border-b-0 last:pb-0"
                  >
                    <div className="font-mono text-xs font-semibold text-emerald-700">
                      {toolItem.name}
                    </div>
                    <p className="mt-1 text-xs text-slate-700 leading-relaxed">
                      {toolItem.purpose}
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      Safeguard: {toolItem.safety}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {activeTab === 'setup' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Local Installation & Execution
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  Run the Streamlit chatbot locally with Python 3.10+
                </p>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                    <Terminal className="w-3.5 h-3.5" aria-hidden="true" />
                    Step 1: Virtual Environment & Dependencies
                  </span>
                  <button
                    type="button"
                    onClick={() =>
                      handleCopy(
                        'cd genai-chatbot\npython -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt',
                        'cmd-install'
                      )
                    }
                    className="text-xs font-medium text-emerald-700 hover:underline"
                  >
                    {copiedId === 'cmd-install' ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <pre className="p-3.5 bg-slate-950 text-slate-100 rounded-lg text-xs overflow-x-auto">
{`cd genai-chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt`}
                </pre>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-slate-700">
                    Step 2: Configure Groq API Key
                  </span>
                  <button
                    type="button"
                    onClick={() =>
                      handleCopy(
                        'cp .env.example .env\n# Edit .env and set GROQ_API_KEY=gsk_...',
                        'cmd-env'
                      )
                    }
                    className="text-xs font-medium text-emerald-700 hover:underline"
                  >
                    {copiedId === 'cmd-env' ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <pre className="p-3.5 bg-slate-950 text-slate-100 rounded-lg text-xs overflow-x-auto">
{`cp .env.example .env
# Generate a key at https://console.groq.com/keys
# Set GROQ_API_KEY="gsk_your_actual_key" inside .env`}
                </pre>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-slate-700">
                    Step 3: Launch Streamlit Application
                  </span>
                  <button
                    type="button"
                    onClick={() => handleCopy('streamlit run app.py', 'cmd-run')}
                    className="text-xs font-medium text-emerald-700 hover:underline"
                  >
                    {copiedId === 'cmd-run' ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <pre className="p-3.5 bg-slate-950 text-slate-100 rounded-lg text-xs overflow-x-auto">
{`streamlit run app.py`}
                </pre>
              </div>
            </section>

            <section className="bg-white border border-slate-200 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-slate-900">
                Troubleshooting Guide
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                Common setup and runtime issues with verified fixes
              </p>

              <div className="mt-5 space-y-4">
                {TROUBLESHOOTING_ITEMS.map((item) => (
                  <div
                    key={item.issue}
                    className="pb-4 border-b border-slate-100 last:border-b-0 last:pb-0"
                  >
                    <div className="text-sm font-semibold text-slate-900">{item.issue}</div>
                    <p className="mt-1 text-xs text-slate-600 leading-relaxed">
                      {item.resolution}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
