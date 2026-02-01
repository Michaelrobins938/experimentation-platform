"use client";

import React, { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
    Area, ReferenceLine, ComposedChart
} from 'recharts';
import {
    Beaker, Shield, FlaskConical, Zap, Info, Clock, Database, Gauge, Play, Pause, RefreshCw, Download, BarChart3,
    Terminal, Activity, Scale, Microscope, Cpu, Radio, Network, GitBranch, AlertCircle, Eye, Target, ZapOff, BookOpen
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Tooltip from '../components/shared/Tooltip';
import InfoPanel from '../components/shared/InfoPanel';

const colorMap = {
    indigo: { border: 'border-l-indigo-500/50', glow: 'bg-indigo-500', marker: 'bg-indigo-500', text: 'text-indigo-400' },
    violet: { border: 'border-l-violet-500/50', glow: 'bg-violet-500', marker: 'bg-violet-500', text: 'text-violet-400' },
    cyan: { border: 'border-l-cyan-500/50', glow: 'bg-cyan-500', marker: 'bg-cyan-500', text: 'text-cyan-400' },
    emerald: { border: 'border-l-emerald-500/50', glow: 'bg-emerald-500', marker: 'bg-emerald-500', text: 'text-emerald-400' },
};

const analysisLogs = [
    {
        id: 1,
        time: '04:12:01',
        event: 'SPRT_KERNEL_CYCLE',
        details: 'Boundary re-computation for interim look 14.',
        icon: Cpu,
        impact: 'Stabilizing always-valid intervals.'
    },
    {
        id: 2,
        time: '04:12:45',
        event: 'SRM_CHECK_SUCCEEDED',
        details: 'Sample ratio balance within 0.1% of target allocation.',
        icon: Shield,
        impact: 'Negating traffic allocation bias.'
    },
    {
        id: 3,
        time: '04:13:30',
        event: 'CUPED_PRE_PERIOD_LINK',
        details: 'Retrieved 30-day pre-experiment behavioral signatures.',
        icon: Database,
        impact: 'Variance reduction engine primed.'
    }
];

export default function ExperimentationDashboard() {
    const [mounted, setMounted] = useState(false);
    const [isRunning, setIsRunning] = useState(true);
    const [sequentialData, setSequentialData] = useState<any[]>([]);
    const [activeAnalysisLog, setActiveAnalysisLog] = useState(0);
    const [investigationMode, setInvestigationMode] = useState(false);

    useEffect(() => {
        setMounted(true);
        const data = Array.from({ length: 20 }, (_, i) => ({
            look: i + 1,
            Upper: 4.5 - (i * 0.15),
            Lower: -4.5 + (i * 0.15),
            ZScore: (Math.sin(i * 0.4) * 2.5) + (Math.random() * 0.3 - 0.15)
        }));
        setSequentialData(data);
    }, []);

    if (!mounted) {
        return <div className="min-h-screen bg-[#0a0a0a]" />;
    }

    const currentLook = 14;
    const currentZScore = sequentialData[currentLook - 1]?.ZScore || 0;
    const upperBound = sequentialData[currentLook - 1]?.Upper || 2.5;

    return (
        <div
            className={`min-h-screen carbon-fiber-bg text-zinc-100 font-mono selection:bg-indigo-500/30 transition-all duration-700 ${investigationMode ? 'hue-rotate-15 saturate-150' : ''}`}
        >
            <div className="scan-line" />

            {/* Header */}
            <header className="relative pt-20 pb-12 px-10">
                <div className="flex justify-between items-end mb-16">
                    <div>
                        <div className="flex items-center gap-3 mb-6">
                            <span className="px-3 py-1 carbon-plate border border-indigo-500/50 text-indigo-500 text-[10px] font-black tracking-[0.3em] uppercase">
                                EXPERIMENT_KERNEL_STABLE
                            </span>
                            <span className="text-zinc-600 text-[10px] font-black tracking-[0.2em] uppercase">MSRPT_VALIDATED</span>
                        </div>
                        <h1 className="text-[10rem] font-black italic tracking-tighter uppercase leading-[0.75] text-white">
                            DECISION <span className="gradient-text">INTEL</span>
                        </h1>
                    </div>
                </div>

                <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 mt-12 bg-black/40 p-6 border border-zinc-800/50 rounded-sm">
                    <div className="flex items-center gap-5">
                        <Tooltip content="THE KERNEL: The primary compute unit. It handles the 'heavy lifting' of statistical transforms, ensuring p-values aren't just numbers, but actionable signals based on rigorous Lan-DeMets alpha spending.">
                            <motion.div
                                whileHover={{ scale: 1.05, rotate: 5 }}
                                className="p-3 bg-indigo-600 text-white rounded-lg cursor-help shadow-lg shadow-indigo-900/30 group"
                            >
                                <FlaskConical size={22} />
                            </motion.div>
                        </Tooltip>
                        <div>
                            <div className="flex items-center gap-3 mb-0.5">
                                <Tooltip content="ALPHA_SPENDING_V2: This engine allows us to 'peek' at the data as it arrives without increasing the chance of a False Positive. Traditional t-tests only allow for one peek at the very end.">
                                    <span className="px-3 py-0.5 bg-zinc-900 border border-zinc-800 text-indigo-400 text-[8px] font-bold tracking-widest uppercase rounded-full flex items-center gap-2 cursor-help group hover:border-indigo-500/30 transition-all">
                                        <Beaker size={9} className="group-hover:animate-pulse" />
                                        ENGINE V2.4_KERNEL
                                    </span>
                                </Tooltip>
                                <Tooltip content="STREAM_INTEGRITY: A live link to our data lake. If this turns amber, there is a delay in event packet processing, which might slightly 'out-date' the current stats.">
                                    <span className="text-[8px] text-zinc-600 font-bold uppercase tracking-wider flex items-center gap-2 cursor-help">
                                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                                        STREAM_LIVE
                                    </span>
                                </Tooltip>
                            </div>
                            <h2 className="text-xl font-black tracking-tight uppercase leading-tight">
                                Stats <span className="text-indigo-400 italic">Command</span> Center
                            </h2>
                            <p className="text-[9px] text-zinc-600 uppercase tracking-widest font-medium">Tactical Experimentation & Decision Intelligence</p>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        <Tooltip content="INVESTIGATION_MODE: Toggles high-contrast mode and highlights technical specifications for deep-drilling into the statistical foundation.">
                            <button
                                onClick={() => setInvestigationMode(!investigationMode)}
                                className={`flex items-center gap-2 px-4 py-3 border text-[10px] font-bold uppercase tracking-widest rounded-lg transition-all ${investigationMode
                                    ? 'border-amber-600 text-amber-400 bg-amber-900/20 shadow-[0_0_15px_rgba(245,158,11,0.2)]'
                                    : 'border-zinc-800 text-zinc-500 bg-zinc-900 hover:border-zinc-700'
                                    }`}
                            >
                                <Eye size={12} />
                                {investigationMode ? 'INVESTIGATION_ON' : 'NORMAL_OPS'}
                            </button>
                        </Tooltip>
                        <Tooltip content="ENGINE_COMMAND: Manually pause the data ingestion. Useful for stabilizing results before a final 'Ship' decision or during site maintenance.">
                            <button
                                onClick={() => setIsRunning(!isRunning)}
                                className={`flex items-center gap-2 px-5 py-3 border text-[10px] font-bold uppercase tracking-widest rounded-lg transition-all ${isRunning
                                    ? 'border-emerald-600 text-emerald-400 bg-emerald-900/20'
                                    : 'border-red-900 text-red-500 bg-red-900/10'
                                    }`}
                            >
                                {isRunning ? <Play size={12} fill="currentColor" /> : <Pause size={12} fill="currentColor" />}
                                {isRunning ? 'ENGINE_LIVE' : 'ENGINE_PAUSED'}
                            </button>
                        </Tooltip>
                        <Tooltip content="DEPLOY_COMMAND: Immediately pushes the variant 'Treatment' to 100% of traffic. WARNING: Should only be executed if Z-Score exceeds boundaries and SRM is within stable limits.">
                            <button className="btn-primary flex items-center gap-2 px-8 py-3 text-white font-bold uppercase tracking-wider text-xs rounded-lg active:scale-95 transition-transform">
                                <Zap size={14} fill="white" />
                                DEPLOY_WINNER
                            </button>
                        </Tooltip>
                    </div>
                </div>
            </header>


            <main className="p-8">
                <div className="max-w-7xl mx-auto space-y-8">

                    {/* Operation Manual */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="relative"
                    >
                        <InfoPanel
                            title="Tactical Operational Protocol"
                            description="Guidelines for Statistical Decision Making"
                            details="This platform is not just a dashboard; it is an observability layer for causality. We use mSPRT (Mixture Sequential Probability Ratio Test) to provide p-values that are always valid, regardless of how often you look at the data."
                            useCase="1. Check for 'SRM' balance (Randomization Health). 2. Monitor 'Sequential Monitoring' for boundary breaks. 3. Look for 'CUPED' efficiency gains. 4. Execute 'DEPLOY' once the mission is clear."
                            technical="Engine utilizes O'Brien-Fleming boundary spending. Covariate adjustment is performed using a shrinkage estimator for theta to minimize pre-period noise bleed-through."
                        />
                        <div className="absolute -top-3 -right-3">
                            <Tooltip content="PROTOCOL_LOCKED: Current analysis protocol is fixed and cannot be altered while the experiment is active to prevent bias contamination.">
                                <div className="bg-zinc-900 border border-zinc-800 p-2 rounded-full cursor-help">
                                    <Shield size={14} className="text-emerald-500" />
                                </div>
                            </Tooltip>
                        </div>
                    </motion.div>

                    {/* Health Metrics */}
                    <section>
                        <div className="flex items-center justify-between mb-5">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-indigo-500/10 rounded-lg">
                                    <Target size={18} className="text-indigo-400" />
                                </div>
                                <h2 className="text-lg font-black uppercase tracking-widest text-zinc-100 italic">Vital Signal Diagnostics</h2>
                            </div>
                            <Tooltip content="A summary of the 'Integrity Score'. 100% means the experiment is perfectly valid and ready for decisioning.">
                                <div className="text-[10px] font-black text-emerald-400 bg-emerald-500/5 px-3 py-1 border border-emerald-500/20 rounded-full tracking-tighter">
                                    SYSTEM_INTEGRITY: 98.4%
                                </div>
                            </Tooltip>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            {[
                                {
                                    label: 'Type I Error (α)', value: '0.05', status: 'LOCKED', color: 'indigo',
                                    tooltip: 'ALPHA: Our risk budget. We accept a 5% chance of being wrong. Lowering this increases the time required for a decision.',
                                    manual: 'Controls the False Positive rate. Always fixed at T=0.'
                                },
                                {
                                    label: 'Statistical Power', value: '82.4%', status: 'OPTIMIZED', color: 'violet',
                                    tooltip: 'POWER: The probability we detect the effect if it exists. At 80%+, we are statistically confident in detecting our target delta.',
                                    manual: 'As Power grows, the chance of a False Negative (missing a winner) shrinks.'
                                },
                                {
                                    label: 'Current MDE', value: '2.14%', status: 'NOMINAL', color: 'cyan',
                                    tooltip: 'MDE: Minimal Detectable Effect. The smallest lift we can see right now. As more users enter the experiment, this gets smaller.',
                                    manual: 'Think of this as the resolution of our microscope. More data = better resolution.'
                                },
                                {
                                    label: 'SRM Test', value: 'p=0.45', status: 'HEALTHY', color: 'emerald',
                                    tooltip: 'SRM (Sample Ratio Mismatch): If Treatment and Control have widely different user counts, something is wrong with the code. p > 0.05 is required.',
                                    manual: 'The ultimate sanity check. If this is red, the experiment results cannot be trusted.'
                                }
                            ].map((metric, i) => {
                                const classes = colorMap[metric.color as keyof typeof colorMap];
                                return (
                                    <Tooltip key={i} content={metric.tooltip}>
                                        <motion.div
                                            whileHover={{ y: -5 }}
                                            className={`stat-card-premium rounded-xl cursor-help border-l-4 ${classes.border} group relative`}
                                        >
                                            <div className={`glow-orb ${classes.glow}`} />
                                            <div className="relative z-10">
                                                <div className="flex items-center justify-between mb-2">
                                                    <div className="flex items-center gap-2">
                                                        <span className={`w-1.5 h-1.5 rounded-full ${classes.marker}`} />
                                                        <span className="text-zinc-500 text-[9px] font-black uppercase tracking-widest">{metric.label}</span>
                                                    </div>
                                                    <Info size={10} className="text-zinc-700 opacity-0 group-hover:opacity-100 transition-opacity" />
                                                </div>
                                                <div className="text-3xl font-black tracking-tight text-white group-hover:text-indigo-400 transition-colors">{metric.value}</div>
                                                <div className="flex items-center justify-between mt-1">
                                                    <div className="text-[9px] text-emerald-400 font-black uppercase flex items-center gap-1">
                                                        <Activity size={8} /> {metric.status}
                                                    </div>
                                                    {investigationMode && (
                                                        <span className="text-[7px] text-zinc-600 font-bold italic">{metric.manual}</span>
                                                    )}
                                                </div>
                                            </div>
                                        </motion.div>
                                    </Tooltip>
                                );
                            })}
                        </div>
                    </section>

                    <div className="grid grid-cols-12 gap-6">
                        {/* Sequential Monitoring Section */}
                        <div className="col-span-12 lg:col-span-8 space-y-6">
                            <section className="tactical-panel p-8 rounded-2xl border-l-4 border-l-indigo-500/30 shadow-volumetric overflow-hidden relative">
                                <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                                    <BarChart3 size={150} />
                                </div>
                                <div className="flex justify-between items-start mb-8 relative z-10">
                                    <div>
                                        <h2 className="text-2xl font-black tracking-tight uppercase mb-1 text-zinc-100 flex items-center gap-3">
                                            <Terminal size={20} className="text-indigo-400" />
                                            Sequential Monitoring
                                        </h2>
                                        <Tooltip content="O'BRIEN-FLEMING: A spending function that makes it very hard to 'stop early' at the beginning of an experiment, but easier at the end. Protective against 'Early Look Bias'.">
                                            <div className="flex items-center gap-2 cursor-help group">
                                                <Shield size={12} className="text-zinc-600 group-hover:text-indigo-400 transition-colors" />
                                                <p className="text-xs text-zinc-500 font-black uppercase tracking-widest">LAN-DEMETS_PROTOCOL_V2</p>
                                            </div>
                                        </Tooltip>
                                    </div>
                                    <div className="status-running px-4 py-2 rounded-lg text-[10px] font-black uppercase tracking-[0.2em] text-emerald-400 bg-emerald-500/5 border border-emerald-500/20">
                                        MONITORING_LIVE
                                    </div>
                                </div>

                                <div className="h-[350px] relative z-10">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={sequentialData}>
                                            <defs>
                                                <linearGradient id="upperGrad" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="0%" stopColor="#dc2626" stopOpacity={0.15} />
                                                    <stop offset="100%" stopColor="#dc2626" stopOpacity={0} />
                                                </linearGradient>
                                                <linearGradient id="lowerGrad" x1="0" y1="1" x2="0" y2="0">
                                                    <stop offset="0%" stopColor="#dc2626" stopOpacity={0.15} />
                                                    <stop offset="100%" stopColor="#dc2626" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.04)" />
                                            <XAxis
                                                dataKey="look"
                                                axisLine={false}
                                                tickLine={false}
                                                tick={{ fill: '#52525b', fontSize: 10, fontWeight: 'bold' }}
                                            />
                                            <YAxis
                                                domain={[-5, 5]}
                                                axisLine={false}
                                                tickLine={false}
                                                tick={{ fill: '#52525b', fontSize: 10, fontWeight: 'bold' }}
                                                label={{ value: 'Z-SCORE_MAGNITUDE', angle: -90, position: 'insideLeft', fill: '#333', fontSize: 10, fontWeight: 'black' }}
                                            />
                                            <RechartsTooltip
                                                cursor={{ stroke: '#6366f1', strokeWidth: 2 }}
                                                contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #222', borderRadius: '8px', fontSize: '11px', fontWeight: 'bold' }}
                                            />
                                            <ReferenceLine y={0} stroke="#333" strokeDasharray="5 5" />
                                            <Area type="stepAfter" dataKey="Upper" stroke="transparent" fill="url(#upperGrad)" />
                                            <Area type="stepAfter" dataKey="Lower" stroke="transparent" fill="url(#lowerGrad)" />
                                            <Line type="stepAfter" dataKey="Upper" stroke="#dc2626" strokeWidth={2} dot={false} strokeDasharray="6 4" />
                                            <Line type="stepAfter" dataKey="Lower" stroke="#dc2626" strokeWidth={2} dot={false} strokeDasharray="6 4" />
                                            <Line type="monotone" dataKey="ZScore" stroke="#6366f1" strokeWidth={3} dot={{ r: 4, fill: '#6366f1', stroke: '#0a0a0a', strokeWidth: 2 }} />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                </div>

                                <div className="mt-8 p-6 glass-surface rounded-xl border-l-4 border-l-indigo-500/30 shadow-volumetric relative z-10 group">
                                    <div className="flex gap-5 items-center">
                                        <div className="p-3 bg-indigo-600 text-white rounded-lg shadow-lg shadow-indigo-900/40 group-hover:rotate-6 transition-transform">
                                            <Info size={20} />
                                        </div>
                                        <div className="flex-1">
                                            <h4 className="text-base font-black uppercase tracking-tight mb-1">INTERIM_ANALYSIS: LOOK {currentLook} OF 20</h4>
                                            <p className="text-zinc-500 text-[11px] font-bold">CURRENT_Z_SCORE: <span className="text-indigo-400">{currentZScore.toFixed(2)}</span> | WIN_THRESHOLD: &gt; +{upperBound.toFixed(2)}</p>
                                        </div>
                                        <Tooltip content="FUTILITY: If even with the best possible future data we can't hit significance, let's stop burning money. This button runs a 'Best Case' simulation.">
                                            <button className="btn-secondary px-6 py-2 font-black uppercase tracking-widest text-[9px] rounded-lg border border-zinc-800 hover:border-indigo-500/30 transition-all hover:bg-indigo-500/5">CHECK_FUTILITY_MODE</button>
                                        </Tooltip>
                                    </div>
                                </div>

                                <div className="mt-8">
                                    <InfoPanel
                                        title="Sequential Intelligence Manual"
                                        description="Understanding the Path to Significance"
                                        details="The center blue line is our Treatment effect. The red boundaries are the 'Decision Fences'. If the blue line hits the TOP, we ship. If it hits the BOTTOM, we kill. Anything in-between is 'Neutral Ground'—continue the mission."
                                        useCase="Surveil daily. We use Alpha Spending to prevent 'P-Hacking' (data-peeking bias). This chart provides a mandate to act the moment significance is mathematical reality."
                                        technical="Computes boundaries via Lan-DeMets alpha spending (O'Brien-Fleming) with 20 preset interim analysis points."
                                    />
                                </div>
                            </section>

                            {/* Logic Console */}
                            <section className="tactical-panel p-8 rounded-2xl border-l-4 border-l-cyan-500/30 shadow-volumetric grad-cyan">
                                <div className="flex items-center justify-between mb-8">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-cyan-500/20 rounded-lg">
                                            <GitBranch size={20} className="text-cyan-400" />
                                        </div>
                                        <h3 className="text-lg font-black uppercase tracking-widest text-white italic">Logic_Inference_Log</h3>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                                        <span className="text-[9px] font-black text-cyan-400 uppercase tracking-widest">REAL_TIME_FEED</span>
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    {analysisLogs.map((item, i) => (
                                        <motion.div
                                            key={item.id}
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: i * 0.1 }}
                                            onMouseEnter={() => setActiveAnalysisLog(i)}
                                            className={`flex gap-4 p-5 rounded-2xl border transition-all cursor-default relative overflow-hidden group ${activeAnalysisLog === i ? 'bg-cyan-500/10 border-cyan-500/30' : 'bg-black/60 border-white/5 hover:border-cyan-500/20'}`}
                                        >
                                            {activeAnalysisLog === i && (
                                                <motion.div
                                                    layoutId="logGlow"
                                                    className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-transparent pointer-events-none"
                                                />
                                            )}
                                            <div className="text-[10px] font-black text-cyan-500/40 group-hover:text-cyan-400 transition-colors uppercase whitespace-nowrap">[{item.time}]</div>
                                            <div className="relative z-10 flex-1">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <item.icon size={12} className="text-cyan-400" />
                                                    <span className="text-xs font-black uppercase text-zinc-100 italic">{item.event}</span>
                                                </div>
                                                <p className="text-[10px] text-zinc-500 leading-relaxed font-bold">{item.details}</p>
                                                {activeAnalysisLog === i && (
                                                    <motion.p
                                                        initial={{ opacity: 0, height: 0 }}
                                                        animate={{ opacity: 1, height: 'auto' }}
                                                        className="text-[9px] text-cyan-400/70 font-black uppercase mt-2 border-t border-cyan-500/20 pt-2 tracking-tighter"
                                                    >
                                                        IMPACT: {item.impact}
                                                    </motion.p>
                                                )}
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>
                                <div className="mt-8">
                                    <InfoPanel
                                        title="Kernel Operations Data"
                                        description="Understanding the System Logic"
                                        details="The Inference Log shows exactly what the Engine is doing at any micro-second. It monitors for data quality, recalculates boundaries, and links pre-experiment behavioral covariates for noise scrubbing."
                                        useCase="Use this as a 'Live Diagnostic'. If the log stops or starts showing 'RETRY', there may be an issue with the underlying data pipeline."
                                        technical="Logged via asynchronous sidecar process to minimize overhead on the primary statistical kernel."
                                    />
                                </div>
                            </section>
                        </div>

                        {/* Sidebar: Advanced Surveillance & Integrity */}
                        <div className="col-span-12 lg:col-span-4 space-y-6">

                            {/* Observability Feed: Nodes */}
                            <section className="tactical-panel p-6 rounded-2xl border-l-4 border-l-purple-500/30 grad-purple shadow-volumetric group">
                                <div className="flex items-center justify-between mb-6">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-purple-500/20 rounded-lg shadow-lg shadow-purple-900/20 group-hover:bg-purple-500/30 transition-colors">
                                            <Activity size={16} className="text-purple-400" />
                                        </div>
                                        <h3 className="text-sm font-black uppercase tracking-[0.1em] text-zinc-100">Operational_Nodes</h3>
                                    </div>
                                    <Tooltip content="All computing nodes are currently healthy and synchronized.">
                                        <span className="text-[10px] font-black text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20 cursor-help">LIVE_HYPERLINK</span>
                                    </Tooltip>
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    {[
                                        { label: 'US_EAST_01', status: 'SYNCD', color: 'bg-emerald-500', latency: '4ms' },
                                        { label: 'EU_WEST_03', status: 'SYNCD', color: 'bg-emerald-500', latency: '32ms' },
                                        { label: 'ASIA_PAC_09', status: 'LATENCY', color: 'bg-amber-500', latency: '150ms' },
                                        { label: 'US_WEST_02', status: 'SYNCD', color: 'bg-emerald-500', latency: '12ms' }
                                    ].map((node, idx) => (
                                        <Tooltip key={idx} content={`NODE_INTEGRITY: Stable. Latency: ${node.latency}. Node is handling 25% of the total event load.`}>
                                            <div className="bg-black/60 border border-white/5 p-3 rounded-xl hover:border-purple-500/40 transition-all cursor-help group/node">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <div className={`w-1.5 h-1.5 rounded-full ${node.color} animate-pulse shadow-[0_0_8px_currentColor]`} />
                                                    <span className="text-[9px] font-black text-zinc-500 group-hover/node:text-zinc-300 transition-colors">{node.label}</span>
                                                </div>
                                                <div className="flex justify-between items-center">
                                                    <div className="text-[10px] font-black text-zinc-100 uppercase tracking-tighter italic">{node.status}</div>
                                                    <span className="text-[8px] font-bold text-zinc-700">{node.latency}</span>
                                                </div>
                                            </div>
                                        </Tooltip>
                                    ))}
                                </div>
                            </section>

                            {/* NOISE_ELIMINATION: CUPED */}
                            <section className="experiment-card p-8 rounded-3xl relative overflow-hidden group shadow-volumetric grad-indigo border-indigo-500/30 shadow-neon-indigo">
                                <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 group-hover:scale-110 group-hover:rotate-12 transition-all duration-700">
                                    <Microscope size={120} />
                                </div>
                                <div className="relative z-10">
                                    <Tooltip content="CUPED: Think of this as 'Noise-Cancelling Headphones' for your data. By looking at how users behaved BEFORE the experiment, we can ignore their natural variation and focus ONLY on what our Treatment changed.">
                                        <h3 className="text-lg font-black uppercase tracking-tighter mb-6 flex items-center gap-3 cursor-help text-indigo-100 italic">
                                            <Zap size={22} className="text-indigo-400 fill-indigo-400/20 shadow-indigo-500/50 shadow-2xl" /> NOISE_ELIMINATION
                                        </h3>
                                    </Tooltip>

                                    <div className="flex items-baseline gap-2 mb-2">
                                        <div className="text-6xl font-black tracking-tightest text-white drop-shadow-[0_0_20px_rgba(99,102,241,0.4)] transition-all group-hover:drop-shadow-[0_0_30px_rgba(99,102,241,0.6)]">35.4</div>
                                        <div className="text-2xl font-black text-indigo-400">%</div>
                                    </div>
                                    <p className="text-zinc-500 text-[10px] font-black tracking-[0.2em] mb-8 uppercase italic">CUPED_SIG_BOOST_INTEL</p>

                                    <div className="space-y-4">
                                        <Tooltip content="This means for every 100 users in the trial, it feels like we have 150. We are essentially 'stealing' data efficiency from history.">
                                            <div className="flex justify-between items-center bg-black/60 border border-white/5 p-4 cursor-help hover:border-indigo-500/40 transition-all rounded-2xl group/item">
                                                <div className="flex flex-col">
                                                    <span className="text-[9px] uppercase font-black tracking-widest text-zinc-500">SAMPLE_ALPHA_GAIN</span>
                                                    <span className="text-[8px] text-indigo-400/60 font-black uppercase tracking-tighter mt-0.5 group-hover/item:text-indigo-400 transition-colors">Efficiency Boost</span>
                                                </div>
                                                <span className="text-2xl font-black tracking-tighter text-white drop-shadow-sm">+1.5x</span>
                                            </div>
                                        </Tooltip>
                                        <Tooltip content="Because we can 'see' significance faster, we can stop the experiment 4 days earlier. No more waiting two weeks for a 2-day result.">
                                            <div className="flex justify-between items-center bg-black/60 border border-white/5 p-4 cursor-help hover:border-indigo-500/40 transition-all rounded-2xl group/item">
                                                <div className="flex flex-col">
                                                    <span className="text-[9px] uppercase font-black tracking-widest text-zinc-500">HORIZON_CONTRACTION</span>
                                                    <span className="text-[8px] text-indigo-400/60 font-black uppercase tracking-tighter mt-0.5 group-hover/item:text-indigo-400 transition-colors">Time Recovery</span>
                                                </div>
                                                <span className="text-2xl font-black tracking-tighter text-white drop-shadow-sm">-4.2D</span>
                                            </div>
                                        </Tooltip>
                                    </div>

                                    <div className="mt-8">
                                        <InfoPanel
                                            title="CUPED Tactical Advantage"
                                            description="Reducing noise to increase speed."
                                            details="CUPED links the pre-experiment period (last 30 days) to the active period. This creates a baseline for every single user, allowing us to scrub out 'natural' variance that would otherwise muddy the result."
                                            useCase="Always keep active. It acts as a safety layer—if a user was already a high-converter before the experiment, CUPED ensures we don't accidentally credit the Treatment for their conversion."
                                            technical="Covariance adjustment using pre-experiment covariates. Theta minimizes var(Y - theta*X). Applied at the micro-segment level."
                                        />
                                    </div>
                                </div>
                            </section>

                            {/* INTEGRITY: Firewall */}
                            <section className="tactical-panel p-8 rounded-3xl grad-emerald border-emerald-500/20 shadow-volumetric shadow-neon-emerald group relative overflow-hidden">
                                <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                                    <Shield size={100} />
                                </div>
                                <Tooltip content="DATA_FIREWALL: A suite of 24/7 monitors that look for corruption. If bots enter the experiment or the hashing algorithm breaks, this turns red.">
                                    <h3 className="text-xs font-black uppercase tracking-[0.2em] text-emerald-400/70 mb-8 flex items-center gap-3 cursor-help group-hover:text-emerald-400 transition-all relative z-10 italic">
                                        <Shield size={16} className="text-emerald-500 fill-emerald-500/10 shadow-emerald-500/50 shadow-2xl" /> INTEGRITY_FIREWALL_PROTOCOL
                                    </h3>
                                </Tooltip>
                                <div className="space-y-8 relative z-10">
                                    {[
                                        {
                                            label: 'TYPE_I_STABILITY', value: '4.8%', status: 'VALID', bar: 96,
                                            tooltip: 'STABILITY: We target <5% false positives. Currently, the kernel is perfectly stable at 4.8%.',
                                            color: 'emerald', sub: 'Calculated False Positive Risk'
                                        },
                                        {
                                            label: 'DATA_QUALITY_SCORE', value: '99.2%', status: 'OPTIMAL', bar: 99,
                                            tooltip: 'QUALITY: 99.2% of users have clean, un-corrupted event history. 0.8% were scrubbed (bots/anomalies).',
                                            color: 'emerald', sub: 'Signal Clarity Index'
                                        },
                                        {
                                            label: 'COVARIATE_EQUILIBRIUM', value: 'p=0.67', status: 'BALANCED', bar: 100,
                                            tooltip: 'BALANCE: Ensures the Control and Treatment groups looked the same BEFORE the experiment started. p > 0.05 is required.',
                                            color: 'emerald', sub: 'Randomization Entropy'
                                        }
                                    ].map((s, idx) => (
                                        <Tooltip key={idx} content={s.tooltip}>
                                            <div className="cursor-help group/item">
                                                <div className="flex justify-between items-end mb-3">
                                                    <div className="flex flex-col">
                                                        <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">{s.label}</span>
                                                        <span className="text-[8px] font-black text-emerald-500/50 uppercase tracking-tighter group-hover/item:text-emerald-400 transition-colors italic">{s.sub}</span>
                                                    </div>
                                                    <span className="text-lg font-black text-zinc-100 tracking-tighter group-hover/item:text-emerald-400 transition-colors">{s.value}</span>
                                                </div>
                                                <div className="h-2 bg-black/60 rounded-full overflow-hidden border border-white/5 shadow-inner">
                                                    <motion.div
                                                        initial={{ width: 0 }}
                                                        animate={{ width: `${s.bar}%` }}
                                                        transition={{ duration: 1.5, delay: idx * 0.2 }}
                                                        className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                                                    />
                                                </div>
                                            </div>
                                        </Tooltip>
                                    ))}
                                </div>
                                <div className="mt-8 relative z-10">
                                    <InfoPanel
                                        title="Integrity Firewall Specs"
                                        description="Maintaining the Statistical Perimeter"
                                        details="The Firewall protects the experiment from 'Garbage In, Garbage Out'. It runs CRC checks on every incoming event packet and monitors the p-value of the covariate balance to ensure randomization hasn't been corrupted by external factors."
                                        useCase="If 'COVARIATE_EQUILIBRIUM' drops, investigate the randomization hash. It usually means an upstream filter is biasing which users enter which group."
                                        technical="Stability monitored via Brownian motion simulation. Covariate p-value uses a Two-Sample Kolmogorov-Smirnov test."
                                    />
                                </div>
                            </section>

                            {/* Live Resource Monitor */}
                            <section className="tactical-panel p-6 rounded-2xl grad-amber border-amber-500/20 shadow-volumetric shadow-neon-amber overflow-hidden relative group">
                                <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                                    <Database size={80} />
                                </div>
                                <div className="flex items-center gap-3 mb-6 relative z-10">
                                    <div className="p-2 bg-amber-500/20 rounded-lg shadow-[0_0_15px_rgba(245,158,11,0.2)]">
                                        <Radio size={16} className="text-amber-400 group-hover:animate-bounce" />
                                    </div>
                                    <h3 className="text-sm font-black uppercase tracking-widest text-zinc-100 italic">Live_Event_Command</h3>
                                </div>
                                <div className="space-y-4 relative z-10">
                                    <Tooltip content="EVENTS_PER_SECOND (EPS): The total number of user actions being processed right now across the entire globe.">
                                        <div className="bg-black/60 p-4 rounded-xl border border-white/5 cursor-help hover:border-amber-500/30 transition-all">
                                            <div className="flex justify-between text-[10px] font-black text-zinc-500 mb-2">
                                                <span className="tracking-widest">INGEST_THROUGHPUT</span>
                                                <span className="text-amber-400">14.2K ACTION/SEC</span>
                                            </div>
                                            <div className="h-1 bg-black/60 rounded-full overflow-hidden">
                                                <div className="h-full bg-gradient-to-r from-amber-600 to-amber-400 w-[65%] animate-pulse shadow-[0_0_10px_rgba(245,158,11,0.3)]" />
                                            </div>
                                        </div>
                                    </Tooltip>
                                    <div className="grid grid-cols-2 gap-3">
                                        <Tooltip content="RETRY_RATE: The percentage of data packets that failed on the first try and are being re-sent. Low is good.">
                                            <div className="text-center p-3 bg-black/60 rounded-xl border border-white/5 hover:border-amber-500/30 transition-all group/stat cursor-help">
                                                <div className="text-[10px] font-black text-zinc-600 leading-none mb-1 group-hover/stat:text-amber-500/70 transition-colors uppercase">RETRY_INDEX</div>
                                                <div className="text-sm font-black text-zinc-100 italic">0.02%</div>
                                            </div>
                                        </Tooltip>
                                        <Tooltip content="BUFFER_LEVEL: How much of our local memory is being used to cache results before they hit the main data store.">
                                            <div className="text-center p-3 bg-black/60 rounded-xl border border-white/5 hover:border-amber-500/30 transition-all group/stat cursor-help">
                                                <div className="text-[10px] font-black text-zinc-600 leading-none mb-1 group-hover/stat:text-amber-500/70 transition-colors uppercase">BUF_LOAD</div>
                                                <div className="text-sm font-black text-zinc-100 italic">12%</div>
                                            </div>
                                        </Tooltip>
                                    </div>
                                </div>
                            </section>
                        </div>
                    </div>

                    {/* Operational Parameters Grid */}
                    <div className="flex items-center gap-3 mt-12 mb-6">
                        <motion.div
                            animate={{ rotate: [0, 90, 0] }}
                            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                        >
                            <Terminal size={18} className="text-zinc-600" />
                        </motion.div>
                        <h2 className="text-lg font-black uppercase tracking-widest text-zinc-500 italic">Operational_Decision_Parameters</h2>
                    </div>
                    <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        {[
                            {
                                icon: FlaskConical, label: 'TEST_KIND', value: 'mSPRT_SEQUENTIAL',
                                tooltip: "TEST_FRAMEWORK: This is the 'math brain' of the mission. mSPRT is the gold standard for continuous testing without False Positive inflation.",
                                tech: 'Always-Valid p-values'
                            },
                            {
                                icon: Target, label: 'NORTH_STAR', value: 'REVENUE_PER_USER',
                                tooltip: "MISSION_KPI: The final value we are trying to change. If this doesn't move, the mission is a wash, regardless of other metrics.",
                                tech: 'Primary Success Vector'
                            },
                            {
                                icon: Clock, label: 'MIN_DURATION', value: '14_DAYS_LOCKED',
                                tooltip: "TIME_LOCK: We must wait 14 days to see a full 'Weekly Cycle'. Users behave differently on weekends than on Mondays.",
                                tech: 'Seasonality Protection'
                            },
                            {
                                icon: Database, label: 'SOURCE_PIPE', value: 'HDFS_GOLD_BATCH',
                                tooltip: "DATA_SOURCE: The vault where the raw actions are stored. This is the finalized batch of truth used for auditing.",
                                tech: 'Immutable Source of Truth'
                            }
                        ].map((item, i) => (
                            <Tooltip key={i} content={item.tooltip}>
                                <motion.div
                                    whileHover={{ scale: 1.02, borderColor: 'rgba(99, 102, 241, 0.4)' }}
                                    className="glass-surface p-6 rounded-2xl flex items-center gap-5 hover:border-indigo-500/30 transition-all cursor-help group shadow-volumetric border border-white/5 bg-[#0d0d0d] relative overflow-hidden"
                                >
                                    <div className="p-3 bg-zinc-900 text-indigo-400 rounded-xl border border-zinc-800 group-hover:bg-indigo-500/10 transition-all duration-300 relative z-10">
                                        <item.icon size={18} />
                                    </div>
                                    <div className="relative z-10">
                                        <div className="text-[10px] uppercase font-black text-zinc-600 tracking-widest mb-0.5 group-hover:text-zinc-400 transition-colors uppercase">{item.label}</div>
                                        <div className="text-sm font-black text-white uppercase tracking-tighter italic">{item.value}</div>
                                        {investigationMode && (
                                            <p className="text-[7px] text-indigo-400/50 font-bold mt-1 uppercase tracking-tight">{item.tech}</p>
                                        )}
                                    </div>
                                    <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-[0.03] transition-opacity">
                                        <item.icon size={60} />
                                    </div>
                                </motion.div>
                            </Tooltip>
                        ))}
                    </section>
                </div>
            </main>

            {/* Tactical Footer */}
            <footer className="p-8 border-t border-zinc-800 bg-[#0a0a0a]/80 backdrop-blur-xl mt-12">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-8">
                    <div className="flex items-center gap-8 text-[10px] font-black text-zinc-700 uppercase tracking-[0.2em]">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse shadow-[0_0_8px_rgba(99,102,241,0.5)]" />
                            <span className="text-indigo-500/70">COMMS_STABLE</span>
                        </div>
                        <Tooltip content="SYSTEM_VERSION: Cumulative updates including SRM auto-detection and CUPED 2.0.">
                            <span className="hover:text-zinc-500 transition-colors cursor-help">SDK_V3.2.1-SECURE</span>
                        </Tooltip>
                        <Tooltip content="All monitoring is compliant with global privacy standards. PII is scrubbed at the edge.">
                            <span className="cursor-help flex items-center gap-2 hover:text-emerald-500 transition-colors">
                                <Shield size={12} /> COMPLIANCE_ENFORCED
                            </span>
                        </Tooltip>
                    </div>
                    <div className="flex gap-8">
                        <Tooltip content="MISSION_EXPORT: Generate a full cryptographic report of every statistical decision made during this analysis window.">
                            <button className="text-[10px] font-black text-zinc-500 hover:text-indigo-400 transition-all uppercase tracking-[0.15em] flex items-center gap-3 group px-4 py-2 bg-white/5 rounded-lg border border-white/5 hover:border-indigo-500/20 active:scale-95">
                                <Download size={12} className="group-hover:translate-y-0.5 transition-transform" />
                                GENERATE_DOSSIER
                            </button>
                        </Tooltip>
                        <span className="text-[10px] font-black text-zinc-800 uppercase tracking-widest italic border-l border-zinc-900 pl-8 flex items-center gap-3 group">
                            <Cpu size={10} className="group-hover:rotate-180 transition-transform duration-1000" />
                            TECH_COMMAND_INTEL // BUILD_829.4
                        </span>
                    </div>
                </div>
            </footer>
        </div>
    );
}
