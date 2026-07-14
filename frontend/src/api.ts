import type { Language, Project } from './types'

const API = import.meta.env.VITE_API_URL ?? ''

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? 'Request failed')
  }
  return response.json() as Promise<T>
}

export const downloadUrl = (path: string) => `${API}${path}`

export async function getLanguages() {
  return json<Language[]>(await fetch(`${API}/api/v1/languages`))
}

export async function getProjects() {
  const result = await json<{ items: Project[] }>(await fetch(`${API}/api/v1/projects`, { cache: 'no-store' }))
  return result.items
}

export async function createProject(
  video: Blob,
  filename: string,
  source: string,
  target: string,
  inputType: 'upload' | 'recording',
  voiceGender: 'female' | 'male',
) {
  const data = new FormData()
  data.append('video', video, filename)
  data.append('source_language', source)
  data.append('target_language', target)
  data.append('input_type', inputType)
  data.append('voice_gender', voiceGender)
  return json<Project>(await fetch(`${API}/api/v1/projects`, { method: 'POST', body: data }))
}

export async function createLiveSession(source: string, target: string, voiceGender: 'female' | 'male') {
  return json<{ id: string; status: string; chunk_count: number }>(await fetch(`${API}/api/v1/live-sessions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_language: source, target_language: target, voice_gender: voiceGender }),
  }))
}

export async function uploadLiveChunk(sessionId: string, index: number, chunk: Blob) {
  const data = new FormData()
  data.append('chunk', chunk, `chunk-${index}.webm`)
  return json(await fetch(`${API}/api/v1/live-sessions/${sessionId}/chunks/${index}`, { method: 'POST', body: data }))
}

export async function finishLiveSession(sessionId: string) {
  return json<Project>(await fetch(`${API}/api/v1/live-sessions/${sessionId}/finish`, { method: 'POST' }))
}

export async function deleteProject(projectId: string) {
  const response = await fetch(`${API}/api/v1/projects/${projectId}`, { method: 'DELETE' })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? 'Could not delete project')
  }
}
