import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, JOB_TERMINAL, type AnalysisJob } from '@/lib/api'

export interface Resource<T> {
  data: T | null
  error: string | null
  loading: boolean
}

/** Загрузка с отменой: результат устаревшего запроса не перетирает актуальный. */
export function useResource<T>(load: () => Promise<T>, key: string): Resource<T> {
  const [state, setState] = useState<Resource<T>>({ data: null, error: null, loading: true })

  useEffect(() => {
    let alive = true
    setState({ data: null, error: null, loading: true })
    load()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e: Error) => alive && setState({ data: null, error: e.message, loading: false }))
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return state
}

const POLL_MS = 500
const RETRY_MS = 2000
const MAX_FAILURES = 5

/**
 * Данные могут принадлежать прежнему id, пока не пришёл ответ для нового: страница сверяет `data.id`.
 * Опрос сохраняемого backend job до терминального состояния. Переживает кратковременный
 * сбой сети (рестарт backend): ошибка показывается после MAX_FAILURES подряд или сразу при 404.
 * `restart` возобновляет опрос после retry.
 */
export function useAnalysisJob(id: string | undefined): Resource<AnalysisJob> & { restart: () => void } {
  const [state, setState] = useState<Resource<AnalysisJob>>({ data: null, error: null, loading: true })
  const [nonce, setNonce] = useState(0)
  const restart = useCallback(() => setNonce((value) => value + 1), [])

  useEffect(() => {
    if (!id) return
    let timer: number
    let alive = true
    let failures = 0

    const poll = async () => {
      try {
        const data = await api.analysisJob(id)
        if (!alive) return
        failures = 0
        setState({ data, error: null, loading: false })
        if (!JOB_TERMINAL.includes(data.status)) timer = window.setTimeout(poll, POLL_MS)
      } catch (error) {
        if (!alive) return
        failures += 1
        const fatal = error instanceof ApiError && error.status === 404
        if (fatal || failures >= MAX_FAILURES) {
          setState({ data: null, error: (error as Error).message, loading: false })
        } else {
          timer = window.setTimeout(poll, RETRY_MS)
        }
      }
    }

    poll()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [id, nonce])

  return { ...state, restart }
}
