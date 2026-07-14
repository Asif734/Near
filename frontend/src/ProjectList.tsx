import { downloadUrl } from './api'
import type { Project } from './types'

type Props = { projects: Project[]; onDelete: (project: Project) => Promise<void> }

export function ProjectList({ projects, onDelete }: Props) {
  return (
    <section className="projects" aria-labelledby="projects-title">
      <div className="section-heading">
        <div><p className="eyebrow">Your workspace</p><h2 id="projects-title">Recent projects</h2></div>
        <span className="count">{projects.length}</span>
      </div>
      <div className="project-grid">
        {projects.length === 0 && <div className="empty">Your translated videos will appear here.</div>}
        {projects.map((project) => (
          <article className="project" key={project.id}>
            <div className={`status-dot ${project.status}`} />
            <div className="project-main">
              <strong title={project.original_filename}>{project.original_filename}</strong>
              <span>{project.source_language} → {project.target_language} · {project.voice_gender} voice · {project.input_type}</span>
              {project.status === 'failed' ? <p className="error">{project.error}</p> : (
                <div className="progress" aria-label={`${project.progress}% complete`}>
                  <span style={{ width: `${project.progress}%` }} />
                </div>
              )}
            </div>
            <div className="project-action">
              <small>{project.stage}</small>
              {project.download_url && <a href={downloadUrl(project.download_url)}>Download</a>}
              {project.artifact_urls && <small className="artifact-links">
                <a href={downloadUrl(project.artifact_urls.deepgram)}>Deepgram JSON</a>{' · '}
                <a href={downloadUrl(project.artifact_urls.segmentation)}>Segments JSON</a>
              </small>}
              <button className="delete-project" disabled={project.status === 'processing'} onClick={() => void onDelete(project)} aria-label={`Delete ${project.original_filename}`}>Delete</button>
            </div>
            <div className="video-comparison">
              <section className="video-pane">
                <div className="video-pane-heading"><strong>Original</strong><a href={downloadUrl(project.input_download_url)}>Download input</a></div>
                <video aria-label={`Original preview of ${project.original_filename}`} src={downloadUrl(project.input_preview_url)} controls playsInline preload="metadata" />
              </section>
              <section className="video-pane">
                <div className="video-pane-heading"><strong>Translated</strong>{project.download_url && <a href={downloadUrl(project.download_url)}>Download output</a>}</div>
                {project.preview_url ? (
                  <video aria-label={`Translated preview of ${project.original_filename}`} src={downloadUrl(project.preview_url)} controls playsInline preload="metadata" />
                ) : <div className="output-placeholder"><span className={`status-dot ${project.status}`} />{project.status === 'failed' ? 'Translation failed' : `${project.stage} · ${project.progress}%`}</div>}
              </section>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}
