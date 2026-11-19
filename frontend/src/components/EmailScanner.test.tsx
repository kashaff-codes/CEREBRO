import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { server } from '../test/msw-handlers';
import userEvent from '@testing-library/user-event';
import { AuthProvider } from '../context/AuthContext';
import { EmailScanner } from './EmailScanner';

vi.mock('../utils/audio', () => ({ playCyberSFX: () => {} }));

const renderWithAuth = (ui: React.ReactNode) => render(<AuthProvider>{ui}</AuthProvider>);

describe('EmailScanner', () => {
  it('renders the forensics panel', async () => {
    renderWithAuth(<EmailScanner />);
    expect(await screen.findByText(/Mail forensics/i)).toBeInTheDocument();
  });

  it('offers to connect Gmail when no inbox is connected', async () => {
    renderWithAuth(<EmailScanner />);
    expect(await screen.findByRole('button', { name: /Connect Gmail/i })).toBeEnabled();
    expect(screen.getByText(/cannot send, delete or move mail/i)).toBeInTheDocument();
  });

  it('scans a connected inbox and shows analytics, riskiest message first', async () => {
    server.use(
      http.get('*/v1/gmail/status', () =>
        HttpResponse.json({
          configured: true,
          connected: true,
          email: 'me@gmail.com',
          messages_total: 1200,
        }),
      ),
    );
    const user = userEvent.setup();
    renderWithAuth(<EmailScanner />);
    expect(await screen.findByText('me@gmail.com')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Scan inbox/i }));
    expect(await screen.findByText('Your account is locked')).toBeInTheDocument();
    expect(screen.getByText('Authentication passed')).toBeInTheDocument();
    expect(screen.getByText('paypa1.com')).toBeInTheDocument();
    // expanding a message shows the signals behind its score
    await user.click(screen.getByRole('button', { name: /Your account is locked/i }));
    expect(await screen.findByText('1 character from paypal')).toBeInTheDocument();
  });

  it('finishes the Google round trip from the return URL and clears it', async () => {
    let body: any = null;
    server.use(
      http.post('*/v1/gmail/connect', async ({ request }) => {
        body = await request.json();
        return HttpResponse.json({ connected: true, email: 'new@gmail.com', messages_total: 5 });
      }),
    );
    window.history.replaceState({}, '', '/?state=1700000000.abc&code=4%2Fxyz&scope=email');
    renderWithAuth(<EmailScanner />);
    expect(await screen.findByText('new@gmail.com')).toBeInTheDocument();
    expect(body).toMatchObject({ code: '4/xyz', state: '1700000000.abc' });
    expect(window.location.search).toBe('');
  });

  it('loads the sample and shows a verdict with indicators (MSW-mocked API)', async () => {
    const user = userEvent.setup();
    renderWithAuth(<EmailScanner />);

    await user.click(screen.getByText(/load sample phishing email/i));
    await user.click(screen.getByRole('button', { name: /Analyse message/i }));

    // A concrete indicator + the verdict from the mocked forensics response.
    // (Exact 'Phishing' avoids also matching the "Load sample phishing email" link.)
    expect(await screen.findByText('LOOKALIKE_SENDER_DOMAIN')).toBeInTheDocument();
    expect(await screen.findByText('Phishing')).toBeInTheDocument();
  });
});
