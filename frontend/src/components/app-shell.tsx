import { useEffect, useState } from 'react'
import {
  BookOpen,
  Check,
  CircleAlert,
  Clock3,
  Menu,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Settings2,
  Sun,
} from 'lucide-react'
import { useTheme } from 'next-themes'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { AsciiLogo } from '@/components/ascii-art'
import { api, type JobStatus } from '@/lib/api'
import { useResource } from '@/lib/hooks'
import { cn } from '@/lib/utils'

const statusIcon: Record<JobStatus, typeof Clock3> = {
  pending: Clock3,
  running: Clock3,
  complete: Check,
  error: CircleAlert,
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('sidebar-collapsed') === 'true')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const history = useResource(api.history, location.pathname)

  useEffect(() => {
    localStorage.setItem('sidebar-collapsed', String(collapsed))
  }, [collapsed])

  return (
    <div className="flex h-svh overflow-hidden bg-background">
      {mobileOpen && (
        <button
          type="button"
          aria-label="Закрыть боковую панель"
          className="fixed inset-0 z-40 bg-black/45 backdrop-blur-[1px] md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex w-[286px] flex-col border-r border-sidebar-border bg-sidebar p-2.5 text-sidebar-foreground transition-transform duration-200 md:relative md:z-20 md:translate-x-0 md:transition-[width]',
          mobileOpen ? 'translate-x-0' : '-translate-x-full',
          collapsed && 'md:w-[68px]',
          !collapsed && 'md:w-[286px]',
        )}
      >
        <div className="flex h-11 items-center gap-2 px-1">
          {collapsed ? (
            <button
              type="button"
              className="group mx-auto hidden size-10 items-center justify-center rounded-lg transition-colors hover:bg-sidebar-accent md:flex"
              aria-label="Развернуть боковую панель"
              title="Развернуть панель"
              onClick={() => setCollapsed(false)}
            >
              <AsciiLogo className="group-hover:hidden" />
              <PanelLeftOpen className="hidden size-4.5 group-hover:block" />
            </button>
          ) : (
            <>
              <Link
                to="/"
                className="hidden min-w-0 items-center gap-2.5 rounded-lg px-2 py-2 hover:bg-sidebar-accent md:flex"
                title="NextWave"
              >
                <AsciiLogo />
                <span className="truncate text-sm">NextWave</span>
              </Link>

              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="ml-auto hidden text-sidebar-foreground md:inline-flex"
                aria-label="Свернуть боковую панель"
                title="Свернуть панель"
                onClick={() => setCollapsed(true)}
              >
                <PanelLeftClose />
              </Button>
            </>
          )}

          <Link
            to="/"
            onClick={() => setMobileOpen(false)}
            className="flex min-w-0 items-center gap-2.5 rounded-lg px-2 py-2 hover:bg-sidebar-accent md:hidden"
            title="NextWave"
          >
            <AsciiLogo />
            <span className="truncate text-sm">NextWave</span>
          </Link>

          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="ml-auto md:hidden"
            aria-label="Закрыть боковую панель"
            onClick={() => setMobileOpen(false)}
          >
            <PanelLeftClose />
          </Button>
        </div>

        <nav className="mt-2 flex flex-col gap-1">
          <SidebarLink
            to="/"
            icon={Plus}
            label="Новое исследование"
            collapsed={collapsed}
            onNavigate={() => setMobileOpen(false)}
            end
          />
          <SidebarLink
            to="/methodology"
            icon={BookOpen}
            label="Методология"
            collapsed={collapsed}
            onNavigate={() => setMobileOpen(false)}
          />
        </nav>

        <div className={cn('mt-6 flex min-h-0 flex-1 flex-col', collapsed && 'md:hidden')}>
          <p className="px-3 pb-2 text-xs font-medium text-sidebar-foreground/50">Сохранённые чаты</p>
          <div className="min-h-0 flex-1 overflow-y-auto pb-4 [scrollbar-width:thin]">
            {history.loading && (
              <div className="space-y-2 px-2">
                {[0, 1, 2].map((item) => (
                  <div key={item} className="h-9 animate-pulse rounded-lg bg-sidebar-accent" />
                ))}
              </div>
            )}

            {history.error && (
              <p className="px-3 py-2 text-xs leading-relaxed text-sidebar-foreground/50">
                История временно недоступна
              </p>
            )}

            {history.data?.length === 0 && (
              <p className="px-3 py-2 text-xs leading-relaxed text-sidebar-foreground/50">
                Завершённые исследования появятся здесь.
              </p>
            )}

            {history.data?.map((item) => {
              const Icon = statusIcon[item.status]
              return (
                <NavLink
                  key={item.id}
                  to={`/analyses/${item.id}`}
                  onClick={() => setMobileOpen(false)}
                  className={({ isActive }) =>
                    cn(
                      'group flex min-h-10 items-center gap-2.5 rounded-lg border border-transparent px-3 py-2 text-sm transition-colors hover:border-sidebar-border',
                      isActive && 'border-sidebar-border bg-sidebar-accent',
                    )
                  }
                >
                  <Icon
                    className={cn(
                      'size-3.5 shrink-0 text-sidebar-foreground/45',
                      item.status === 'running' && 'animate-pulse',
                    )}
                  />
                  <span className="min-w-0 flex-1 truncate">{item.query}</span>
                </NavLink>
              )
            })}
          </div>
        </div>

        <div className="relative mt-auto border-t border-sidebar-border pt-2">
          {settingsOpen && <ThemeMenu collapsed={collapsed} close={() => setSettingsOpen(false)} />}
          <button
            type="button"
            className={cn(
              'flex h-11 w-full items-center gap-3 rounded-lg px-3 text-left text-sm transition-colors hover:bg-sidebar-accent',
              settingsOpen && 'bg-sidebar-accent',
              collapsed && 'md:justify-center md:px-0',
            )}
            aria-expanded={settingsOpen}
            onClick={() => setSettingsOpen((value) => !value)}
            title="Настройки"
          >
            <Settings2 className="size-4.5 shrink-0" />
            <span className={cn(collapsed && 'md:hidden')}>Настройки</span>
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center border-b border-border px-3 md:hidden">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Открыть боковую панель"
            onClick={() => setMobileOpen(true)}
          >
            <Menu />
          </Button>
          <Link to="/" className="ml-2 flex items-center gap-2 text-sm">
            <AsciiLogo className="size-7" /> NextWave
          </Link>
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto"
            nativeButton={false}
            render={<Link to="/methodology" />}
          >
            Методология
          </Button>
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto scroll-smooth">{children}</main>
      </div>
    </div>
  )
}

function SidebarLink({
  to,
  icon: Icon,
  label,
  collapsed,
  onNavigate,
  end = false,
}: {
  to: string
  icon: typeof Plus
  label: string
  collapsed: boolean
  onNavigate: () => void
  end?: boolean
}) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onNavigate}
      title={collapsed ? label : undefined}
      className={({ isActive }) =>
        cn(
          'flex h-11 items-center gap-3 rounded-lg border border-transparent px-3 text-sm transition-colors hover:border-sidebar-border',
          isActive && 'border-sidebar-border bg-sidebar-accent',
          collapsed && 'md:justify-center md:px-0',
        )
      }
    >
      <Icon className="size-4.5 shrink-0" />
      <span className={cn(collapsed && 'md:hidden')}>{label}</span>
    </NavLink>
  )
}

function ThemeMenu({ collapsed, close }: { collapsed: boolean; close: () => void }) {
  const { theme, setTheme } = useTheme()
  const options = [
    { value: 'light', label: 'Светлая', icon: Sun },
    { value: 'dark', label: 'Тёмная', icon: Moon },
    { value: 'system', label: 'Как в системе', icon: Settings2 },
  ]

  return (
    <div
      className={cn(
        'absolute bottom-12 left-0 z-20 w-full min-w-56 rounded-xl border border-border bg-popover p-1.5 text-popover-foreground shadow-xl',
        collapsed && 'md:left-12 md:w-56',
      )}
    >
      <p className="px-2.5 py-2 text-xs font-medium text-muted-foreground">Оформление</p>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm hover:bg-muted"
          onClick={() => {
            setTheme(option.value)
            close()
          }}
        >
          <option.icon className="size-4" />
          <span>{option.label}</span>
          {theme === option.value && <Check className="ml-auto size-4" />}
        </button>
      ))}
    </div>
  )
}
