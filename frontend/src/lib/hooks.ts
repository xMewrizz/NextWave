import { useEffect, useState } from 'react'
import { api, TERMINAL, type Analysis } from '@/lib/api'

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

/** Опрос статуса анализа до терминального состояния. */
export function useAnalysis(id: string | undefined): Resource<Analysis> {
  const [state, setState] = useState<Resource<Analysis>>({ data: null, error: null, loading: true })

  useEffect(() => {
    if (!id) return
    let timer: number
    let alive = true

    const poll = async () => {
      try {
        const data = await api.analysis(id)
        if (!alive) return
        setState({ data, error: null, loading: false })
        // Короткий опрос нужен только демонстрационному контуру; production job использует SSE.
        if (!TERMINAL.includes(data.status)) timer = setTimeout(poll, 400)
      } catch (e) {
        if (alive) setState({ data: null, error: (e as Error).message, loading: false })
      }
    }

    poll()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [id])

  return state
}
