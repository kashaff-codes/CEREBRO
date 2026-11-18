import { test, expect } from '@playwright/test';

// Runs on both Desktop Chrome and an emulated iPhone (see playwright.config.ts),
// which guards the mobile rendering path that the iOS auth fix depended on.
test('login screen renders', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByPlaceholder(/analyst@/i)).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Sign in' })).toBeVisible();
  await expect(page.getByText(/Verification intelligence/i)).toBeVisible();
});

test('can switch to the sign-up tab', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Create account' }).click();
  await expect(page.getByLabel('Full name')).toBeVisible();
});

test('the sign-in fields are labelled', async ({ page }) => {
  // getByLabel fails if the label/control association breaks, which is the
  // accessibility bug worth catching.
  await page.goto('/');
  await expect(page.getByLabel('Email')).toBeVisible();
  await expect(page.getByLabel('Password')).toBeVisible();
});
