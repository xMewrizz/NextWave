import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

const RAMP = ' .·:-=+*%#@'
const NOISE = '01<>/\\|+*#%&$@'

type Mouse = { x: number; y: number; on: number }
// Яркость клетки 0..1. x, y — пиксели внутри холста; w, h — его размер; t — мс.
type Field = (x: number, y: number, w: number, h: number, t: number, m: Mouse) => number

// Живой ASCII-арт: холст заполняет родителя, каждая клетка — символ по яркости поля.
// У курсора символы «перемешиваются» и светлеют. Цвет — --foreground.
function AsciiField({ field, cell = 10, fps = 30, className }: { field: Field; cell?: number; fps?: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current!
    const ctx = canvas.getContext('2d')!
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches
    const m = { x: -1e4, y: -1e4, on: 0, target: 0 }
    const cw = cell * 0.62
    let w = 0
    let h = 0
    let frame = 0
    let last = 0

    function draw(t: number) {
      m.on += (m.target - m.on) * 0.1
      ctx.clearRect(0, 0, w, h)
      ctx.fillStyle = getComputedStyle(canvas).getPropertyValue('--foreground')
      const tick = Math.floor(t / 90)
      for (let y = cell / 2; y < h; y += cell) {
        for (let x = cw / 2; x < w; x += cw) {
          const v = field(x, y, w, h, t, m)
          if (v < 0.05) continue
          const near = m.on * Math.exp(-((Math.hypot(x - m.x, y - m.y) / (cell * 6)) ** 2))
          const hash = Math.abs(Math.sin(x * 12.99 + y * 78.23 + tick) * 43758.5) % 1
          const ch = near > 0.35 && hash < near ? NOISE[Math.floor(hash * 997) % NOISE.length] : RAMP[Math.min(RAMP.length - 1, Math.ceil(v * (RAMP.length - 1)))]
          ctx.globalAlpha = Math.min(1, 0.2 + v * 0.8 + near * 0.6)
          ctx.fillText(ch, x, y)
        }
      }
      ctx.globalAlpha = 1
    }

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = canvas.clientWidth
      h = canvas.clientHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.font = `${cell}px ui-monospace, SFMono-Regular, Menlo, monospace`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      draw(last)
    }
    const onMove = (e: PointerEvent) => {
      const box = canvas.getBoundingClientRect()
      m.x = e.clientX - box.left
      m.y = e.clientY - box.top
      const pad = cell * 4
      m.target = m.x > -pad && m.y > -pad && m.x < w + pad && m.y < h + pad ? 1 : 0
      if (still) {
        m.on = m.target
        draw(last)
      }
    }
    const onLeave = () => (m.target = 0)
    const loop = (t: number) => {
      frame = requestAnimationFrame(loop)
      if (t - last < 1000 / fps) return
      last = t
      draw(t)
    }

    const observer = new ResizeObserver(resize)
    observer.observe(canvas)
    window.addEventListener('pointermove', onMove)
    document.addEventListener('pointerleave', onLeave)
    // ponytail: fillText на каждую клетку ~30 fps; атлас глифов или WebGL — если глобус начнёт тормозить на слабых ноутбуках
    if (!still) frame = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      window.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerleave', onLeave)
    }
  }, [field, cell, fps])

  return <canvas ref={ref} aria-hidden="true" className={cn('block', className)} />
}

// cn здесь не сливает классы Tailwind, поэтому размер по умолчанию — только если className не передан.
// Логотип «N»: две стойки и диагональ. При наведении по букве бежит волна.
const logoField: Field = (x, y, w, h, t, m) => {
  const u = x / w
  const v = y / h
  const n = u < 0.24 || u > 0.76 || Math.abs(v - u) < 0.2 ? 1 : 0
  return n * (0.75 + 0.25 * m.on * Math.sin(t * 0.012 - (u + v) * 9))
}

export function AsciiLogo({ className }: { className?: string }) {
  return <AsciiField field={logoField} cell={4.5} className={cn('shrink-0', className ?? 'size-8')} />
}

// Глобус: процедурные «континенты» + контурные линии океана, свет сверху-слева, яркий край.
// Курсор поворачивает планету по горизонтали.
const globeField: Field = (x, y, w, h, t, m) => {
  const R = w * 0.9 // шире холста: края планеты уходят за экран, как в макете
  const dx = (x - w / 2) / R
  const dy = (y - (h * 0.5 + R)) / R
  const d2 = dx * dx + dy * dy
  if (d2 > 1) return Math.max(0, 0.45 - (Math.sqrt(d2) - 1) * 14) // атмосфера
  const z = Math.sqrt(1 - d2)
  const a = t * 0.00006 + m.on * (Math.min(Math.max(m.x / w, 0), 1) - 0.5) * 0.9
  const px = dx * Math.cos(a) + z * Math.sin(a)
  const pz = -dx * Math.sin(a) + z * Math.cos(a)
  const py = dy
  const n =
    Math.sin(3.1 * px + 1.7 * py + 0.3) * Math.sin(2.3 * py - 2.9 * pz + 0.5) +
    0.5 * Math.sin(5.3 * pz + 4.1 * px) * Math.sin(4.7 * py + 1.3) +
    0.25 * Math.sin(11 * px - 9 * pz + 7 * py)
  const light = Math.max(0, -0.45 * dx - 0.7 * dy + 0.55 * z)
  const rim = Math.pow(1 - z, 4) // край — самое яркое, как в макете
  if (n > 0.22) return Math.min(1, 0.22 + light * 0.3 + rim)
  const contour = Math.abs(Math.sin(n * 22)) < 0.2 ? 0.2 : 0.05
  return Math.min(1, contour * (0.5 + light) + rim)
}

export function AsciiGlobe({ className }: { className?: string }) {
  return <AsciiField field={globeField} cell={8} className={className ?? 'h-80 w-full'} />
}
