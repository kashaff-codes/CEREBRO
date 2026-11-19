import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AuthProvider } from '../context/AuthContext';
import { CyberMonitor } from './CyberMonitor';

vi.mock('../utils/audio', () => ({ playCyberSFX: () => {} }));

const renderPage = () =>
  render(
    <AuthProvider>
      <CyberMonitor />
    </AuthProvider>,
  );

describe('CyberMonitor (Network watch)', () => {
  it('shows the measured accuracy, including the unflattering held-out number', async () => {
    renderPage();
    expect(await screen.findByText(/measured, not claimed/i)).toBeInTheDocument();
    expect(screen.getByText('99.7%')).toBeInTheDocument(); // seen days
    expect(screen.getByText('25.3%')).toBeInTheDocument(); // unseen days
    expect(screen.queryByText(/Autoencoder/)).not.toBeInTheDocument();
  });

  it('replays labelled traffic with the true answer beside each verdict', async () => {
    renderPage();
    expect(await screen.findByText(/not your network/i)).toBeInTheDocument();
    expect(await screen.findByText('Really was')).toBeInTheDocument();
  });

  it('lists live feed status', async () => {
    renderPage();
    expect(await screen.findByText('abuse.ch Feodo Tracker')).toBeInTheDocument();
  });

  it('scores an uploaded Zeek log and surfaces the threat-feed match', async () => {
    const user = userEvent.setup();
    renderPage();
    const input = (await screen.findByTestId('flow-file')) as HTMLInputElement;
    await user.upload(input, new File(['#fields\tts\n'], 'conn.log', { type: 'text/plain' }));
    expect(await screen.findByText('Critical')).toBeInTheDocument();
    const table = await screen.findByText(/Highest-risk flows/);
    // the flagged row carries the feed badge for its destination address
    expect(within(table.parentElement!).getByText('abuse.ch Feodo Tracker')).toBeInTheDocument();
  });
});
