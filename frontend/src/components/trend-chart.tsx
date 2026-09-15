import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import type { TimelinePoint } from '@/lib/api'
import { formatPeriod } from '@/lib/format'

const config = {
  documents: { label: 'Документов', color: 'var(--chart-1)' },
  share: { label: 'Доля в направлении, %', color: 'var(--chart-2)' },
} satisfies ChartConfig

const withShare = (timeline: TimelinePoint[]) =>
  timeline.map((point) => ({ ...point, share: +(point.share * 100).toFixed(2) }))

export function TrendSparkline({ timeline }: { timeline: TimelinePoint[] }) {
  return (
    <ChartContainer config={config} className="h-10 w-24">
      <AreaChart data={timeline} margin={{ top: 2, bottom: 2, left: 0, right: 0 }}>
        <Area
          dataKey="documents"
          type="monotone"
          stroke="var(--color-documents)"
          fill="var(--color-documents)"
          fillOpacity={0.15}
          strokeWidth={1.5}
          isAnimationActive={false}
          dot={false}
        />
      </AreaChart>
    </ChartContainer>
  )
}

export function TrendChart({ timeline }: { timeline: TimelinePoint[] }) {
  return (
    <ChartContainer config={config} className="h-64 w-full">
      <AreaChart data={withShare(timeline)} margin={{ left: -12, right: 8, top: 8 }}>
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis
          dataKey="period"
          tickLine={false}
          axisLine={false}
          tickMargin={8}
          tickFormatter={formatPeriod}
        />
        <YAxis tickLine={false} axisLine={false} width={48} allowDecimals={false} />
        <ChartTooltip
          content={<ChartTooltipContent labelFormatter={(label) => formatPeriod(String(label))} />}
        />
        <Area
          dataKey="documents"
          type="monotone"
          stroke="var(--color-documents)"
          fill="var(--color-documents)"
          fillOpacity={0.18}
          strokeWidth={2}
        />
        <Area
          dataKey="share"
          type="monotone"
          stroke="var(--color-share)"
          fill="var(--color-share)"
          fillOpacity={0.08}
          strokeWidth={2}
          strokeDasharray="4 4"
        />
      </AreaChart>
    </ChartContainer>
  )
}
