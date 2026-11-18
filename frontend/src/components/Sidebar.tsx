import React, { useState } from 'react';
import { View } from '../types';
import { useAuth } from '../context/AuthContext';
import { playCyberSFX, toggleMute, getMuteStatus } from '../utils/audio';
import {
  LayoutGrid,
  ScanSearch,
  Activity,
  Mail,
  UserRound,
  Volume2,
  VolumeX,
  LogOut,
} from 'lucide-react';

interface SidebarProps {
  currentView: View;
  setCurrentView: (view: View) => void;
}

const ITEMS = [
  { id: View.DASHBOARD, no: '00', label: 'Overview', icon: LayoutGrid },
  { id: View.NEWS_SCANNER, no: '01', label: 'Verify a claim', icon: ScanSearch },
  { id: View.CYBER_MONITOR, no: '02', label: 'Network watch', icon: Activity },
  { id: View.EMAIL_SCANNER, no: '03', label: 'Mail forensics', icon: Mail },
  { id: View.PROFILE, no: '04', label: 'Your file', icon: UserRound },
];

/** The CEREBRO mark: an aperture, closed to a slit. Drawn, not an icon font. */
export const Mark: React.FC<{ size?: number; animate?: boolean }> = ({ size = 28, animate }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
    <circle cx="16" cy="16" r="15" fill="none" stroke="#C9A45C" strokeWidth="1.2" />
    <circle cx="16" cy="16" r="10" fill="none" stroke="rgba(236,230,217,0.35)" strokeWidth="0.8" />
    <path
      d="M6 16 Q16 8 26 16 Q16 24 6 16 Z"
      fill="none"
      stroke="#ECE6D9"
      strokeWidth="1"
      className={animate ? 'animate-draw' : undefined}
      style={{ ['--len' as string]: 60 }}
    />
    <circle cx="16" cy="16" r="2.4" fill="#C9A45C" />
  </svg>
);

export const Sidebar: React.FC<SidebarProps> = ({ currentView, setCurrentView }) => {
  const { user, logout } = useAuth();
  const [muted, setMuted] = useState(getMuteStatus());

  return (
    <aside className="flex h-screen w-[248px] shrink-0 flex-col border-r border-slate-900 bg-[#0E0E11]">
      <div className="flex items-center gap-3 px-6 pb-8 pt-8">
        <Mark />
        <div>
          <p className="font-display text-[19px] tracking-[0.18em] text-slate-100">CEREBRO</p>
          <p className="file-no -mt-0.5">verification desk</p>
        </div>
      </div>

      <nav className="flex-1 px-3" aria-label="Workspaces">
        <p className="label-caps px-3 pb-3">Files</p>
        <ul className="space-y-0.5">
          {ITEMS.map(({ id, no, label, icon: Icon }) => {
            const active = currentView === id;
            return (
              <li key={id}>
                <button
                  onClick={() => {
                    setCurrentView(id);
                  }}
                  onMouseEnter={() => playCyberSFX('hover')}
                  aria-current={active ? 'page' : undefined}
                  className={`group relative flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors ${
                    active ? 'bg-slate-900 text-slate-100' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <span
                    className={`absolute left-0 top-2 bottom-2 w-[2px] rounded-full transition-colors ${
                      active ? 'bg-brass-400' : 'bg-transparent'
                    }`}
                  />
                  <Icon
                    className={`h-[17px] w-[17px] ${active ? 'text-brass-300' : 'text-slate-600 group-hover:text-slate-400'}`}
                    strokeWidth={1.6}
                  />
                  <span className="flex-1 text-[14px]">{label}</span>
                  <span className="file-no opacity-70">{no}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="border-t border-slate-900 px-6 py-5">
        <div className="mb-4 flex items-center justify-between">
          <span className="label-caps">Sound</span>
          <button
            onClick={() => setMuted(toggleMute())}
            className="text-slate-500 transition-colors hover:text-brass-300"
            aria-label={muted ? 'Turn sound on' : 'Turn sound off'}
          >
            {muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </button>
        </div>
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-[13px] text-slate-200">{user?.name}</p>
            <p className="truncate text-[11px] text-slate-600">{user?.email}</p>
          </div>
          <button
            onClick={() => logout()}
            className="rounded-md p-1.5 text-slate-500 transition-colors hover:bg-slate-900 hover:text-slate-200"
            aria-label="Sign out"
          >
            <LogOut className="h-4 w-4" strokeWidth={1.6} />
          </button>
        </div>
      </div>
    </aside>
  );
};
