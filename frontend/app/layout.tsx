import type { Metadata } from "next";
import { MotionConfig } from "framer-motion";
import { Header } from "@/components/Header";
import { ApiStatus } from "@/components/ApiStatus";
import "./globals.css";

export const metadata: Metadata = {
  title: "HOUSEPRICE ML",
  description: "Predicción inteligente de precios inmobiliarios",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es">
      <body>
        <MotionConfig reducedMotion="user">
          <div className="app-background relative min-h-screen overflow-hidden">
            <div className="blob blob-1 left-[-10%] top-[-12%] h-96 w-96 bg-indigo-600" />
            <div className="blob blob-2 right-[-8%] top-[25%] h-[26rem] w-[26rem] bg-purple-600" />
            <div className="blob blob-3 bottom-[-18%] left-[25%] h-80 w-80 bg-sky-600" />

            <div className="relative z-10 mx-auto flex max-w-5xl items-start justify-between gap-4 px-4 pb-8 pt-10">
              <Header />
              <ApiStatus />
            </div>

            <main className="relative z-10 mx-auto max-w-5xl px-4 pb-16">
              {children}
            </main>
          </div>
        </MotionConfig>
      </body>
    </html>
  );
}
