import React from 'react';
import '../styles/index.css';

export const metadata = {
  title: 'PanGNN: Topological Foundation Model for Structural Variant Imputation',
  description: 'An interactive publication-grade dashboard showcasing GNN pathway calculations and ClinVar mapping on Directed Acyclic Pangenome Graphs.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
