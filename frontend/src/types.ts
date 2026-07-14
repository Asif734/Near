export type Language = { code: string; name: string; voice: string }

export type Project = {
  id: string
  original_filename: string
  input_type: 'upload' | 'recording'
  source_language: string
  target_language: string
  voice_gender: 'female' | 'male'
  status: 'queued' | 'processing' | 'completed' | 'failed'
  stage: string
  progress: number
  error: string | null
  created_at: string
  updated_at: string
  download_url: string | null
  preview_url: string | null
  input_download_url: string
  input_preview_url: string
  artifact_urls: Record<string, string> | null
}
