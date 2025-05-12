# UAI PDF Processor Frontend

This is the frontend application for the Privacy-First Document Processor, built with Next.js, TypeScript, and Tailwind CSS.

## Features

- PDF document upload and viewing
- Manual redaction with rectangle selection
- Automatic PII detection and redaction
- Combined manual and automatic redaction
- Multi-language translation support (English ↔ Indonesian)

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create a `.env.local` file in the root directory with the following content:
```
NEXT_PUBLIC_API_URL=http://localhost:8080
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

## Project Structure

- `app/` - Next.js app router pages and layouts
- `components/` - React components
  - `DocumentUploader.tsx` - File upload and processing mode selection
  - `PDFViewer.tsx` - PDF viewing and redaction interface
  - `LoadingSpinner.tsx` - Loading indicator
- `services/` - API services and utilities
  - `api.ts` - Backend API communication

## Development

- Run tests: `npm test`
- Build for production: `npm run build`
- Start production server: `npm start`

## Environment Variables

- `NEXT_PUBLIC_API_URL` - Backend API URL (default: http://localhost:8080)

## Dependencies

- Next.js - React framework
- PDF.js - PDF rendering
- pdf-lib - PDF manipulation
- react-dropzone - File upload
- Tailwind CSS - Styling
- TypeScript - Type safety

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
