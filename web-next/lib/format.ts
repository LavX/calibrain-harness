// Accuracy rules: exchange price strings are shown exactly as delivered (no float re-rounding);
// unknown values render NA, never 0; direction always carries a sign, never colour alone.
// Every time on screen is UTC and says so; ages are relative ("20 h ago").

export type Num = number | string | null | undefined;

/** The one token for an unknown value, everywhere (U+2014). */
export const NA = String.fromCharCode(0x2014);

export const isNum = (x: Num): x is number | string =>
  x !== null && x !== undefined && x !== "" && Number.isFinite(Number(x));

export const fx = (x: Num, nd = 2): string =>
  isNum(x) ? Number(x).toLocaleString("en-US", { minimumFractionDigits: nd, maximumFractionDigits: nd }) : NA;

export const sg = (x: Num, nd = 2, suf = ""): string => {
  if (!isNum(x)) return NA;
  const n = Number(x);
  return (n > 0 ? "+" : n < 0 ? "−" : "") + fx(Math.abs(n), nd) + suf;
};

export const ud = (x: Num): string => (isNum(x) ? (Number(x) > 0 ? "up" : Number(x) < 0 ? "dn" : "") : "");

export const usd = (x: Num): string => {
  if (!isNum(x)) return NA;
  const n = Number(x), a = Math.abs(n);
  const s = a >= 1e9 ? (a / 1e9).toFixed(2) + "B" : a >= 1e6 ? (a / 1e6).toFixed(2) + "M" : a >= 1e3 ? (a / 1e3).toFixed(1) + "K" : a.toFixed(0);
  return (n < 0 ? "−" : "") + s;
};

/** Exchange price string -> thousands separators, original decimals preserved. */
export const px = (s: Num): string => {
  if (!isNum(s)) return NA;
  const [i, f] = String(s).split(".");
  return Number(i).toLocaleString("en-US") + (f !== undefined ? "." + f : "");
};

/** A float price from a status file (not an exchange string): 8 significant digits, no float noise, thousands separators. */
export const pf = (x: Num): string => {
  if (!isNum(x)) return NA;
  const n = Number(Number(x).toPrecision(8));
  if (Math.abs(n) < 1e-6) return String(n);
  const t = n.toFixed(Math.min(20, Math.max(0, 8 - Math.floor(Math.log10(Math.abs(n) || 1)) - 1))); // toFixed accepts 0..100; >= 1e8 needs 0
  return px(t.includes(".") ? t.replace(/0+$/, "").replace(/\.$/, "") : t);
};

export const decimalsOf = (s: Num): number => (String(s ?? "").split(".")[1] || "").length;

const D = (ts: number) => new Date(ts * 1000);
const UTC = { timeZone: "UTC", hourCycle: "h23" } as const;
/** "14:05 UTC" */
export const hhmm = (ts: number) => D(ts).toLocaleTimeString("en-GB", { ...UTC, hour: "2-digit", minute: "2-digit" }) + " UTC";
/** "14:05:09 UTC" */
export const hhmmss = (ts: number) => D(ts).toLocaleTimeString("en-GB", { ...UTC, hour: "2-digit", minute: "2-digit", second: "2-digit" }) + " UTC";
/** "20 Nov 14:05 UTC" */
export const dt = (ts: number) => D(ts).toLocaleString("en-GB", { ...UTC, day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }).replace(",", "") + " UTC";
/** A duration in seconds: "40 s", "12 min", "20 h", "3 d". */
export const age = (s: Num): string => {
  if (!isNum(s)) return NA;
  const n = Math.max(0, Number(s));
  return n < 1 ? "<1 s" : n < 60 ? `${n | 0} s` : n < 3600 ? `${(n / 60) | 0} min` : n < 172800 ? `${(n / 3600) | 0} h` : `${(n / 86400) | 0} d`;
};
/** "20 h ago" for a unix time in seconds. */
export const ago = (ts: Num): string => (isNum(ts) ? `${age(Date.now() / 1000 - Number(ts))} ago` : NA);
/** Money: 2 decimals, thousands separators, the currency. */
export const money = (x: Num, ccy = "USDT", nd = 2): string => (isNum(x) ? `${fx(x, nd)} ${ccy}` : NA);
/** Signed money: "+12.34 USDT". */
export const smoney = (x: Num, ccy = "USDT", nd = 2): string => (isNum(x) ? `${sg(x, nd)} ${ccy}` : NA);
export const hrs = (m: number) => (m >= 60 ? `${m / 60}h` : `${m}m`);
export const short = (sym: string) => sym.replace("USDT", "");
export const TFL: Record<string, string> = { "1": "1M", "5": "5M", "15": "15M", "60": "1H", "240": "4H" };
export const SERIES_C = ["--s1", "--s2", "--s3", "--s4", "--s5"];

export const cssVar = (v: string): string =>
  typeof window === "undefined" ? "" : getComputedStyle(document.documentElement).getPropertyValue(v).trim();
