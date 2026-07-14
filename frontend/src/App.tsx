import { useEffect, useState } from 'react'
import { createProject, deleteProject, getLanguages, getProjects } from './api'
import { ProjectList } from './ProjectList'
import { Recorder } from './Recorder'
import type { Language, Project } from './types'

type Input = { blob: Blob; filename: string; type: 'upload' | 'recording' }

export function App() {
  const [mode, setMode] = useState<'upload' | 'recording'>('upload')
  const [input, setInput] = useState<Input | null>(null)
  const [languages, setLanguages] = useState<Language[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [source, setSource] = useState('bn')
  const [target, setTarget] = useState('en')
  const [voiceGender, setVoiceGender] = useState<'female' | 'male'>('female')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [previewUrl, setPreviewUrl] = useState('')

  useEffect(() => {
    if (!input || input.type !== 'upload') { setPreviewUrl(''); return }
    const url = URL.createObjectURL(input.blob)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [input])

  function chooseFile(file?: File) {
    setDragging(false)
    setError('')
    if (!file) return
    if (!file.type.startsWith('video/')) { setError('Drop a valid video file.'); return }
    setInput({ blob: file, filename: file.name, type: 'upload' })
  }

  async function refresh() {
    try { setProjects(await getProjects()) } catch (reason) { setError((reason as Error).message) }
  }

  useEffect(() => {
    void getLanguages().then(setLanguages).catch((reason: Error) => setError(reason.message))
    void refresh()
    const timer = window.setInterval(() => void refresh(), 3000)
    return () => window.clearInterval(timer)
  }, [])

  const active = projects.some((item) => item.status === 'queued' || item.status === 'processing')

  async function submit() {
    if (!input || source === target) return
    setBusy(true); setError('')
    try {
      const project = await createProject(input.blob, input.filename, source, target, input.type, voiceGender)
      setProjects((items) => [project, ...items])
      setInput(null)
    } catch (reason) { setError((reason as Error).message) } finally { setBusy(false) }
  }

  async function remove(project: Project) {
    if (!window.confirm(`Delete “${project.original_filename}” and all generated files?`)) return
    setError('')
    try {
      await deleteProject(project.id)
      setProjects((items) => items.filter((item) => item.id !== project.id))
    } catch (reason) { setError((reason as Error).message) }
  }

  return (
    <main>
      <nav><div className="brand"><span>◈</span> Dubflow</div><div className="worker-state"><i className={active ? 'pulse' : ''} />{active ? 'Processing' : 'Ready'}</div></nav>
      <header>
        <p className="eyebrow">AI-powered video localization</p>
        <h1>Your voice,<br /><em>every language.</em></h1>
        <p className="lede">Upload a video or record one here. We prepare it in the background while you keep working.</p>
      </header>

      <section className="studio">
        <div className="tabs" role="tablist">
          <button className={mode === 'upload' ? 'selected' : ''} onClick={() => { setMode('upload'); setInput(null) }}>↑ Upload video</button>
          <button className={mode === 'recording' ? 'selected' : ''} onClick={() => { setMode('recording'); setInput(null) }}>● Record here</button>
        </div>

        {mode === 'upload' ? (
          <label
            className={`dropzone ${dragging ? 'dragging' : ''} ${previewUrl ? 'has-preview' : ''}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
            onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy' }}
            onDragLeave={(event) => { event.preventDefault(); if (event.currentTarget === event.target) setDragging(false) }}
            onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]) }}
          >
            <input type="file" accept="video/*" onChange={(event) => {
              chooseFile(event.target.files?.[0])
            }} />
            {previewUrl ? <video aria-label="Selected video preview" src={previewUrl} controls playsInline onClick={(event) => event.preventDefault()} /> : <span className="upload-icon">↑</span>}
            <strong>{input?.type === 'upload' ? input.filename : dragging ? 'Drop video here' : 'Drag and drop or choose a video'}</strong>
            <small>{input ? 'Click anywhere outside the controls to replace it' : 'MP4, MOV or WebM · up to 500 MB'}</small>
          </label>
        ) : <Recorder source={source} target={target} voiceGender={voiceGender} onError={setError} onProject={(project) => setProjects((items) => [project, ...items])} />}

        <div className="settings">
          <label>Spoken language<select value={source} onChange={(event) => setSource(event.target.value)}>{languages.map((lang) => <option value={lang.code} key={lang.code}>{lang.name}</option>)}</select></label>
          <span className="arrow">→</span>
          <label>Translate into<select value={target} onChange={(event) => setTarget(event.target.value)}>{languages.map((lang) => <option value={lang.code} key={lang.code}>{lang.name}</option>)}</select></label>
          <label>Voice<select value={voiceGender} onChange={(event) => setVoiceGender(event.target.value as 'female' | 'male')}><option value="female">Female</option><option value="male">Male{target === 'en' ? ' · Odysseus' : ''}</option></select></label>
          {mode === 'upload' && <button className="submit" disabled={!input || source === target || busy} onClick={submit}>{busy ? 'Uploading…' : 'Start translation →'}</button>}
        </div>
        {source === target && <p className="error">Choose two different languages.</p>}
        {error && <p className="error">{error}</p>}
      </section>
      <ProjectList projects={projects} onDelete={remove} />
      <footer>Dubflow <span>Background processing · private workspace</span></footer>
    </main>
  )
}
