const nextConfig = {
    reactStrictMode: true,
    async headers() {
        return [
            {
                source: '/(.*)',
                headers: [
                    {
                        key: 'Content-Security-Policy',
                        value: [
                            "default-src 'self'",
                            "script-src 'self' 'unsafe-eval' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.tailwindcss.com",
                            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.tailwindcss.com",
                            "img-src 'self' data: blob: https://cdn.jsdelivr.net https://cdn.tailwindcss.com",
                            "font-src 'self' data:",
                            "connect-src 'self'",
                            "media-src 'self'",
                            "object-src 'none'",
                            "base-uri 'self'",
                            "form-action 'self'",
                            "frame-ancestors 'none'",
                            "upgrade-insecure-requests"
                        ].join('; ')
                    }
                ]
            }
        ];
    }
};

module.exports = nextConfig;
