import React from 'react'
import clsx from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...args: any[]) {
  return twMerge(clsx(args))
}

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className={cn('rounded-2xl bg-card-900/70 border border-border-700 shadow-soft backdrop-blur-sm', className)}>
      {children}
    </div>
  )
}

export function CardHeader({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn('px-5 py-4 border-b border-border-700', className)}>{children}</div>
}

export function CardContent({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn('px-5 py-4', className)}>{children}</div>
}

export function Button(
  { className, variant = 'primary', size = 'md', ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
    size?: 'sm' | 'md'
  }
) {
  const base = 'inline-flex items-center justify-center rounded-xl font-medium transition focus:outline-none focus:ring-2 focus:ring-sky-500/40 disabled:opacity-50 disabled:cursor-not-allowed'
  const variants: Record<string, string> = {
    primary: 'bg-sky-500/90 hover:bg-sky-500 text-white',
    secondary: 'bg-white/8 hover:bg-white/12 text-white border border-border-700',
    ghost: 'bg-transparent hover:bg-white/8 text-white',
    danger: 'bg-rose-500/90 hover:bg-rose-500 text-white'
  }
  const sizes: Record<string, string> = {
    sm: 'h-9 px-3 text-sm',
    md: 'h-10 px-4 text-sm'
  }
  return <button className={cn(base, variants[variant], sizes[size], className)} {...props} />
}

export function Input({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        'h-10 w-full rounded-xl bg-white/5 border border-border-700 px-3 text-sm text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-sky-500/40',
        className
      )}
      {...props}
    />
  )
}

export function Textarea({ className, ...props }: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        'min-h-[96px] w-full rounded-xl bg-white/5 border border-border-700 px-3 py-2 text-sm text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-sky-500/40',
        className
      )}
      {...props}
    />
  )
}

export function Select({ className, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        'h-10 w-full rounded-xl bg-white/5 border border-border-700 px-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-sky-500/40',
        className
      )}
      {...props}
    >
      {children}
    </select>
  )
}

export function Badge({ className, tone = 'neutral', children }: { className?: string; tone?: 'neutral' | 'good' | 'warn' | 'bad'; children: React.ReactNode }) {
  const tones: Record<string, string> = {
    neutral: 'bg-white/8 text-white/80 border border-border-700',
    good: 'bg-emerald-500/15 text-emerald-200 border border-emerald-500/30',
    warn: 'bg-amber-500/15 text-amber-200 border border-amber-500/30',
    bad: 'bg-rose-500/15 text-rose-200 border border-rose-500/30'
  }
  return <span className={cn('inline-flex items-center rounded-lg px-2 py-1 text-xs', tones[tone], className)}>{children}</span>
}

export function Divider({ className }: { className?: string }) {
  return <div className={cn('h-px bg-white/10', className)} />
}
