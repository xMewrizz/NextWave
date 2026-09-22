import { FileText, Network, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { TrendSparkline } from '@/components/trend-chart'
import { trendScore, type Bucket, type Trend } from '@/lib/api'
import { formatPeriod, percent } from '@/lib/format'
import { cn } from '@/lib/utils'

const borders: Record<Bucket, string> = {
  main: 'border-foreground',
  watchlist: 'border-muted-foreground',
  excluded: 'border-border',
}

export function TrendCard({ trend, to }: { trend: Trend; to: string }) {
  const border = borders[trend.status]
  return (
    <Link to={to} className="group block">
      <Card className="transition-all group-hover:-translate-y-0.5 group-hover:ring-foreground/30 group-hover:shadow-md">
        <CardContent className="flex gap-4">
          <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full bg-foreground text-sm font-medium tabular-nums text-background">
            {trend.rank}
          </span>

          <div className="min-w-0 flex-1">
            <h3 className="font-medium text-balance group-hover:underline group-hover:underline-offset-4">
              {trend.canonical_name}
            </h3>
            <p className="mt-1 text-sm text-pretty text-muted-foreground">{trend.summary}</p>

            <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
              <Badge variant="secondary">
                <Sparkles />
                оценка модели {percent(trendScore(trend))}
              </Badge>
              {trend.document_count !== null && <Badge variant="outline">
                <FileText />
                {trend.document_count} документов
              </Badge>}
              <Badge variant="outline">
                <Network />
                {trend.features.independent_source_count} независимых
              </Badge>
              {trend.first_seen && <span className="ml-1">первое упоминание {formatPeriod(trend.first_seen)}</span>}
            </div>

            <p className={cn('mt-3 border-l-2 pl-3 text-xs text-pretty text-muted-foreground', border)}>
              {trend.explanation}
            </p>
          </div>

          {trend.timeline.length > 0 && <div className="hidden shrink-0 self-center sm:block">
            <TrendSparkline timeline={trend.timeline} />
          </div>}
        </CardContent>
      </Card>
    </Link>
  )
}
