import { useEffect, useRef, useState } from 'react'
import { createLiveSession, finishLiveSession, uploadLiveChunk } from './api'
import type { Project } from './types'

type Props = {
  source: string
  target: string
  voiceGender: 'female' | 'male'
  onProject: (project: Project) => void
  onError: (message: string) => void
}

function preferredMimeType() {
  const types = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
  return types.find((type) => MediaRecorder.isTypeSupported(type)) ?? ''
}

export function Recorder({ source, target, voiceGender, onProject, onError }: Props) {
  const preview = useRef<HTMLVideoElement>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const chunks = useRef<Blob[]>([])
  const uploads = useRef<Promise<unknown>[]>([])
  const sessionId = useRef('')
  const chunkIndex = useRef(0)
  const playbackUrl = useRef<string | null>(null)
  const [state, setState] = useState<'idle' | 'ready' | 'starting' | 'recording' | 'finalizing' | 'done'>('idle')
  const [seconds, setSeconds] = useState(0)
  const [uploaded, setUploaded] = useState(0)

  useEffect(() => {
    if (state !== 'recording') return
    const timer = window.setInterval(() => setSeconds((value) => value + 1), 1000)
    return () => window.clearInterval(timer)
  }, [state])

  useEffect(() => () => {
    stream.current?.getTracks().forEach((track) => track.stop())
    if (playbackUrl.current) URL.revokeObjectURL(playbackUrl.current)
  }, [])

  async function enableCamera() {
    onError('')
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      if (preview.current) {
        preview.current.src = ''
        preview.current.controls = false
        preview.current.srcObject = stream.current
        await preview.current.play()
      }
      setState('ready')
    } catch {
      onError('Camera and microphone permission is required to record.')
    }
  }

  async function start() {
    if (!stream.current || source === target) return
    setState('starting'); onError('')
    try {
      const session = await createLiveSession(source, target, voiceGender)
      sessionId.current = session.id
      chunks.current = []; uploads.current = []; chunkIndex.current = 0
      setUploaded(0); setSeconds(0)
      const mimeType = preferredMimeType()
      const mediaRecorder = new MediaRecorder(stream.current, mimeType ? { mimeType } : undefined)
      recorder.current = mediaRecorder
      mediaRecorder.ondataavailable = (event) => {
        if (!event.data.size) return
        chunks.current.push(event.data)
        const index = chunkIndex.current++
        const upload = uploadLiveChunk(sessionId.current, index, event.data)
          .then(() => setUploaded((value) => value + 1))
        uploads.current.push(upload)
      }
      mediaRecorder.onstop = () => void finalize(mediaRecorder.mimeType)
      mediaRecorder.start(10_000)
      setState('recording')
    } catch (reason) {
      setState('ready'); onError((reason as Error).message)
    }
  }

  async function finalize(mimeType: string) {
    setState('finalizing')
    try {
      await Promise.all(uploads.current)
      const blob = new Blob(chunks.current, { type: mimeType || 'video/webm' })
      if (playbackUrl.current) URL.revokeObjectURL(playbackUrl.current)
      playbackUrl.current = URL.createObjectURL(blob)
      if (preview.current) {
        preview.current.srcObject = null
        preview.current.src = playbackUrl.current
        preview.current.controls = true
      }
      const project = await finishLiveSession(sessionId.current)
      onProject(project)
      stream.current?.getTracks().forEach((track) => track.stop())
      stream.current = null
      setState('done')
    } catch (reason) {
      setState('done'); onError((reason as Error).message)
    }
  }

  function stop() {
    if (recorder.current?.state === 'recording') {
      recorder.current.requestData()
      recorder.current.stop()
    }
  }

  return <div className="recorder">
    <video ref={preview} muted={state !== 'done'} playsInline aria-label="Camera and recording preview" />
    <div className="recorder-controls">
      {state === 'idle' && <button className="secondary" onClick={enableCamera}>Enable camera</button>}
      {state === 'ready' && <button className="record" onClick={start} disabled={source === target}><span />Start recording</button>}
      {state === 'starting' && <button className="record" disabled>Starting session…</button>}
      {state === 'recording' && <button className="record active" onClick={stop}><span />Stop · {seconds}s · {uploaded} chunks uploaded</button>}
      {state === 'finalizing' && <button className="record" disabled>Finalizing recording…</button>}
      {state === 'done' && <button className="secondary" onClick={enableCamera}>Record again</button>}
    </div>
  </div>
}
