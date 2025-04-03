import '../styles/globals.css';
import type { AppProps } from 'next/app';
import Head from 'next/head';
import Layout from '../components/layout/Layout';

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>NeuraShield</title>
        <meta name="description" content="NeuraShield cybersecurity platform" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Layout>
        <Component {...pageProps} />
      </Layout>
    </>
  );
}

export default MyApp; 