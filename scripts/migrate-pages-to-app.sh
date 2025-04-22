#!/bin/bash

# Script to help migrate from Next.js Pages Router to App Router
# This script preserves both approaches during transition to maintain local testing

set -e

echo "Starting Pages to App Router migration..."

# Create app directory structure if it doesn't exist
mkdir -p frontend/app

# For each page in the pages directory, create equivalent in app directory
echo "Creating equivalent app router files for existing pages..."

for page in $(find frontend/pages -name "*.tsx" -o -name "*.jsx" -o -name "*.js" -not -path "*/api/*" -not -path "*/_*"); do
  # Extract the page name without extension and path
  pagename=$(basename $page | cut -d. -f1)
  
  # Skip if it's a special Next.js file
  if [[ "$pagename" == "_app" || "$pagename" == "_document" ]]; then
    continue
  fi
  
  # Create the equivalent path in app directory
  if [[ "$pagename" == "index" ]]; then
    target_dir="frontend/app"
  else
    target_dir="frontend/app/$pagename"
    mkdir -p $target_dir
  fi
  
  # Create the page.tsx file with import from the original page
  echo "Creating $target_dir/page.tsx..."
  
  # Only create if it doesn't exist yet
  if [[ ! -f "$target_dir/page.tsx" ]]; then
    echo "import { default as PageComponent } from '../../pages/$pagename';

// This is a bridge component that allows both routers to work during migration
export default function Page() {
  return <PageComponent />;
}

// Metadata
export const metadata = {
  title: '${pagename^} - NeuraShield',
  description: 'NeuraShield cybersecurity platform',
};
" > "$target_dir/page.tsx"
  fi
done

# Create the layout.tsx file in the app directory if it doesn't exist
if [[ ! -f "frontend/app/layout.tsx" ]]; then
  echo "Creating app/layout.tsx..."
  echo "import '../styles/globals.css';

export const metadata = {
  title: 'NeuraShield',
  description: 'Comprehensive cybersecurity platform with AI and blockchain technology',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang=\"en\">
      <body>{children}</body>
    </html>
  )
}
" > "frontend/app/layout.tsx"
fi

echo "Migration bridge created. You can now run the application with either Pages or App Router."
echo "To fully migrate, gradually move logic from pages/* to app/*/ and update imports."
echo "When ready for production, remove the pages directory and update next.config.mjs." 