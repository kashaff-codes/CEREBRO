import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { playCyberSFX } from '../utils/audio';
import { Mark } from './Sidebar';

/**
 * Sign in / create an account.
 *
 * Design notes, because the previous screen is the cautionary tale: it framed
 * the form with a fake event log ("IP 192.168.0.1 blocked"), a colour-theme
 * picker, a "decryption" animation and labels like CRYPTO ENDPOINT EMAIL.
 * It read as costume. This one is a single, symmetric column on black: the
 * aperture mark opens, the name is set in a high-contrast serif with wide
 * tracking, and the form uses plain words. The restraint is the point —
 * the product is judged on whether people trust what it tells them.
 */
export const Auth: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, signup } = useAuth();

  const switchMode = (toLogin: boolean) => {
    setIsLogin(toLogin);
    setError('');
    playCyberSFX('click');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (!email || !password) {
      setError('Enter your email and password.');
      return;
    }
    if (!isLogin && !name) {
      setError('Tell us your name.');
      return;
    }
    setLoading(true);
    try {
      if (isLogin) await login(email, password);
      else await signup(name, email, password);
      playCyberSFX('success');
    } catch (err: any) {
      playCyberSFX('alarm');
      setError(err?.message || 'That did not work. Check the details and try again.');
    } finally {
      setLoading(false);
    }
  };

  const input =
    'w-full rounded-lg border border-slate-800 bg-[#101013] px-4 py-3 text-[15px] text-slate-100 ' +
    'placeholder:text-slate-600 outline-none transition-colors focus:border-brass-600';

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-slate-950 px-6 py-16">
      {/* Concentric rings: the barrel-sight motif, drawn once, very quietly. */}
      <svg
        className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 animate-iris"
        width="1100"
        height="1100"
        viewBox="0 0 1100 1100"
        aria-hidden="true"
      >
        {[540, 430, 330, 240].map((r, i) => (
          <circle
            key={r}
            cx="550"
            cy="550"
            r={r}
            fill="none"
            stroke="rgba(236,230,217,1)"
            strokeOpacity={0.035 + i * 0.012}
          />
        ))}
        <g className="animate-sweep" style={{ transformOrigin: '550px 550px' }}>
          <line x1="550" y1="10" x2="550" y2="70" stroke="#C9A45C" strokeOpacity="0.35" />
        </g>
      </svg>

      <div className="relative w-full max-w-[400px] animate-rise">
        <div className="mb-10 flex flex-col items-center text-center">
          <Mark size={56} animate />
          <h1 className="mt-6 font-display text-[38px] tracking-[0.32em] text-slate-100 pl-[0.32em]">
            CEREBRO
          </h1>
          <div className="mt-3 flex items-center gap-3">
            <span className="h-px w-8 bg-brass-700" />
            <p className="label-caps">Verification intelligence</p>
            <span className="h-px w-8 bg-brass-700" />
          </div>
          <p className="mt-5 max-w-[320px] text-[15px] leading-relaxed text-slate-400">
            Paste what you have read. CEREBRO finds the sources, weighs them, and shows its working.
          </p>
        </div>

        <div className="panel p-7">
          <div
            role="tablist"
            aria-label="Account"
            className="mb-7 grid grid-cols-2 rounded-lg border border-slate-800 p-1"
          >
            {[
              ['Sign in', true],
              ['Create account', false],
            ].map(([label, value]) => (
              <button
                key={label as string}
                role="tab"
                type="button"
                aria-selected={isLogin === value}
                onClick={() => switchMode(value as boolean)}
                className={`rounded-md py-2 text-[13px] font-medium transition-colors ${
                  isLogin === value
                    ? 'bg-slate-900 text-slate-100'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {label as string}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            {!isLogin && (
              <div className="animate-rise">
                <label htmlFor="auth-name" className="label-caps mb-2 block">
                  Full name
                </label>
                <input
                  id="auth-name"
                  type="text"
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Jane Bond"
                  className={input}
                />
              </div>
            )}
            <div>
              <label htmlFor="auth-email" className="label-caps mb-2 block">
                Email
              </label>
              <input
                id="auth-email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="analyst@company.com"
                className={input}
              />
            </div>
            <div>
              <label htmlFor="auth-password" className="label-caps mb-2 block">
                Password
              </label>
              <input
                id="auth-password"
                type="password"
                autoComplete={isLogin ? 'current-password' : 'new-password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className={input}
              />
            </div>

            {error && (
              <p
                role="alert"
                className="rounded-lg border border-signal-800 bg-signal-950/40 px-4 py-3 text-[13px] text-signal-200"
              >
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="group relative w-full overflow-hidden rounded-lg bg-brass-400 py-3 text-[14px] font-semibold tracking-wide text-[#16120A] transition-colors hover:bg-brass-300 disabled:opacity-60"
            >
              {loading ? 'One moment…' : isLogin ? 'Sign in' : 'Create account'}
            </button>
          </form>
        </div>

        <p className="mt-8 text-center file-no">
          {isLogin ? 'No account yet? ' : 'Already have one? '}
          <button
            type="button"
            onClick={() => switchMode(!isLogin)}
            className="text-brass-300 underline decoration-brass-800 underline-offset-4 hover:text-brass-200"
          >
            {isLogin ? 'Create one' : 'Sign in'}
          </button>
        </p>
      </div>
    </div>
  );
};
