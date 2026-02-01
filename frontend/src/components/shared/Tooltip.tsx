"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface TooltipProps {
    children: React.ReactNode;
    content: React.ReactNode;
    position?: 'top' | 'bottom' | 'left' | 'right';
    width?: string;
}

const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-3',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-3',
    left: 'right-full top-1/2 -translate-y-1/2 mr-3',
    right: 'left-full top-1/2 -translate-y-1/2 ml-3',
};

export default function Tooltip({ children, content, position = 'top', width = 'w-72' }: TooltipProps) {
    const [isVisible, setIsVisible] = useState(false);

    return (
        <div
            className="relative inline-block w-full"
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
        >
            {children}
            <AnimatePresence>
                {isVisible && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.96, y: position === 'top' ? 6 : -6 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.96, y: position === 'top' ? 6 : -6 }}
                        transition={{ duration: 0.15, ease: 'easeOut' }}
                        className={`absolute z-[100] pointer-events-none ${positionClasses[position]} ${width}`}
                    >
                        <div className="relative p-4 bg-[#161616] border border-zinc-700/50 rounded-lg shadow-xl shadow-black/40 backdrop-blur-sm">
                            {/* Accent Bar */}
                            <div className="absolute left-0 top-0 bottom-0 w-[3px] bg-gradient-to-b from-indigo-500 to-violet-500 rounded-l-lg" />

                            {/* Header */}
                            <div className="text-[9px] font-bold uppercase tracking-widest text-indigo-400 mb-2 ml-2 flex items-center gap-2">
                                <div className="w-1.5 h-1.5 bg-indigo-500 rounded-full" />
                                Context
                            </div>

                            {/* Content */}
                            <div className="text-[11px] leading-relaxed text-zinc-300 font-medium ml-2">
                                {content}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
