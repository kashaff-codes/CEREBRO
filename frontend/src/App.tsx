import React, { useState, useEffect } from 'react';
import { View } from './types';
import { Sidebar } from './components/Sidebar';
import { DashboardHome } from './components/DashboardHome';
import { NewsScanner } from './components/NewsScanner';
import { CyberMonitor } from './components/CyberMonitor';
import { EmailScanner } from './components/EmailScanner';
import { Auth } from './components/Auth';
import { Profile } from './components/Profile';
import { useAuth } from './context/AuthContext';
import { playCyberSFX } from './utils/audio';
import { motion, AnimatePresence } from 'motion/react';

/** Each workspace is a "file": a number, a name, and one line on what it does. */
const FILES: Record<View, { no: string; title: string; brief: string }> = {
  [View.DASHBOARD]: {
    no: '00',
    title: 'Overview',
    brief: 'What CEREBRO has examined, and what it found.',
  },
  [View.NEWS_SCANNER]: {
    no: '01',
    title: 'Verify a claim',
    brief: 'Paste anything you have read. It is researched, sourced and scored.',
  },
  [View.CYBER_MONITOR]: {
    no: '02',
    title: 'Network watch',
    brief:
      'Your logs scored by detectors trained on 2.8 million real flows, and checked against live threat feeds.',
  },
  [View.EMAIL_SCANNER]: {
    no: '03',
    title: 'Mail forensics',
    brief: 'Headers, authentication and links, read the way an analyst would.',
  },
  [View.PROFILE]: {
    no: '04',
    title: 'Your file',
    brief: 'Account, history and preferences.',
  },
};

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<View>(View.DASHBOARD);
  const { user, loading } = useAuth();

  useEffect(() => {
    if (user) playCyberSFX('click');
  }, [currentView]);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center gap-6">
        <svg width="72" height="72" viewBox="0 0 72 72" aria-hidden="true">
          <circle cx="36" cy="36" r="34" fill="none" stroke="rgba(236,230,217,0.12)" />
          <circle
            cx="36"
            cy="36"
            r="34"
            fill="none"
            stroke="#C9A45C"
            strokeWidth="1.5"
            strokeDasharray="40 174"
            className="animate-sweep"
            style={{ transformOrigin: '36px 36px' }}
          />
        </svg>
        <p className="label-caps">Opening the file</p>
      </div>
    );
  }

  if (!user) return <Auth />;

  const file = FILES[currentView];

  const renderContent = () => {
    switch (currentView) {
      case View.NEWS_SCANNER:
        return <NewsScanner />;
      case View.CYBER_MONITOR:
        return <CyberMonitor />;
      case View.EMAIL_SCANNER:
        return <EmailScanner />;
      case View.PROFILE:
        return <Profile />;
      default:
        return <DashboardHome systemColor="blue" />;
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans">
      <Sidebar currentView={currentView} setCurrentView={setCurrentView} />

      <main className="flex-1 h-screen overflow-y-auto custom-scrollbar">
        <div className="mx-auto w-full max-w-[1180px] px-10 py-10">
          <header className="mb-10 flex items-end justify-between gap-8">
            <div className="animate-rise">
              <p className="file-no mb-3">FILE {file.no}</p>
              <h1 className="font-display text-[44px] leading-[1.05] text-slate-100">
                {file.title}
              </h1>
              <p className="mt-3 max-w-xl text-[15px] text-slate-400">{file.brief}</p>
            </div>
            <button
              onClick={() => setCurrentView(View.PROFILE)}
              className="group flex items-center gap-3 rounded-full border border-slate-800 py-1.5 pl-1.5 pr-4 transition-colors hover:border-brass-700"
              aria-label="Open your profile"
            >
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-900 font-display text-sm text-brass-300">
                {user.name.charAt(0).toUpperCase()}
              </span>
              <span className="text-left">
                <span className="block text-[13px] font-medium text-slate-200">{user.name}</span>
                <span className="block text-[11px] text-slate-500">Analyst</span>
              </span>
            </button>
          </header>

          <div className="hairline mb-10" />

          <AnimatePresence mode="wait">
            <motion.div
              key={currentView}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.35, ease: [0.2, 0.7, 0.2, 1] }}
            >
              {renderContent()}
            </motion.div>
          </AnimatePresence>

          <footer className="mt-16 flex items-center justify-between border-t border-slate-900 pt-6 text-[11px] text-slate-600">
            <span>CEREBRO · verification intelligence</span>
            <span className="file-no">Every score shows its sources.</span>
          </footer>
        </div>
      </main>
    </div>
  );
};

export default App;
