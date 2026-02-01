"use client";

import React, { useState } from 'react';
import { ChevronDown, Layers, Cpu, Target, FlaskConical } from 'lucide-react';

interface InfoPanelProps {
    title: string;
    description: string;
    details?: string;
    useCase?: string;
    technical?: string;
}

export default function InfoPanel({ title, description, details, useCase, technical }: InfoPanelProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className="tactical-panel mt-6 overflow-hidden rounded-xl">
            {/* Accent Bar */}
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-indigo-500 via-violet-500 to-purple-500 opacity-50" />

            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full p-5 flex items-center justify-between text-left relative z-10 hover:bg-white/[0.02] transition-colors"
            >
                <div className="flex items-center gap-4">
                    <div className="w-10 h-10 glass-surface flex items-center justify-center rounded-lg">
                        <FlaskConical size={18} className="text-indigo-400" />
                    </div>
                    <div>
                        <h4 className="text-sm font-bold uppercase tracking-wide text-zinc-200">
                            {title}
                        </h4>
                        <p className="text-[10px] text-zinc-500 font-medium mt-0.5 uppercase tracking-wider">
                            {description}
                        </p>
                    </div>
                </div>
                <div className={`p-2 rounded-full transition-all duration-300 ${isExpanded ? 'rotate-180 text-indigo-400 bg-indigo-500/10' : 'text-zinc-600'}`}>
                    <ChevronDown size={16} />
                </div>
            </button>

            {isExpanded && (
                <div className="px-5 pb-6 pt-3 border-t border-white/5">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Statistical Context */}
                        <div className="p-4 glass-surface rounded-lg border-l-2 border-l-cyan-500/40">
                            <div className="flex items-center gap-2 mb-3">
                                <Layers size={12} className="text-cyan-400" />
                                <h5 className="text-[10px] font-bold uppercase tracking-wider text-cyan-400">Statistical Context</h5>
                            </div>
                            <p className="text-[11px] leading-relaxed text-zinc-400">
                                {details || "Sequential testing framework using O'Brien-Fleming alpha spending functions for controlled interim analyses."}
                            </p>
                        </div>

                        {/* Decision Logic */}
                        <div className="p-4 glass-surface rounded-lg border-l-2 border-l-indigo-500/40">
                            <div className="flex items-center gap-2 mb-3">
                                <Target size={12} className="text-indigo-400" />
                                <h5 className="text-[10px] font-bold uppercase tracking-wider text-indigo-400">Decision Logic</h5>
                            </div>
                            <p className="text-[11px] leading-relaxed text-zinc-400">
                                {useCase || "Monitor boundaries to determine early stopping for significance or futility, reducing time-to-decision by up to 40%."}
                            </p>
                        </div>

                        {/* Implementation */}
                        <div className="p-4 glass-surface rounded-lg border-l-2 border-l-emerald-500/40">
                            <div className="flex items-center gap-2 mb-3">
                                <Cpu size={12} className="text-emerald-400" />
                                <h5 className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">Implementation</h5>
                            </div>
                            <p className="text-[11px] leading-relaxed text-zinc-400">
                                {technical || "mSPRT with always-valid confidence intervals. CUPED variance reduction achieves 35%+ sample efficiency."}
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
