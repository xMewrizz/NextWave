import { NavLink } from 'react-router-dom'
import { cn } from '@/lib/utils'

const links = [
  { to: '/', label: 'Поиск' },
  { to: '/methodology', label: 'Методология' },
]

export function NavHeader() {
  return (
    <header className="border-b bg-gradient-to-r from-accent/60 to-transparent">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <span className="font-semibold text-primary">Радар зарождающихся технологий</span>
        <nav className="flex gap-4">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) =>
                cn('text-sm text-muted-foreground hover:text-foreground', isActive && 'font-medium text-foreground')
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
