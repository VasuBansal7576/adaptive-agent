import { useState } from "react";
import type { ConsoleTransport, EnvironmentPackageForm } from "../api/transport";
import { REQUIRED_PACKAGE_FIELDS } from "../api/transport";
import type { EnvironmentPackageSummary } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";
import { Banner, EmptyState, LoadingState } from "../components/ui";

const EMPTY_FORM: EnvironmentPackageForm = {
  environmentId: "",
  version: "",
  toolSchemas: "",
  policyRef: "",
  evaluatorRef: "",
  resetRef: "",
};

export function RegistryView({
  transport,
  environments,
  loading,
}: {
  transport: ConsoleTransport;
  environments: EnvironmentPackageSummary[];
  loading: boolean;
}) {
  const [form, setForm] = useState<EnvironmentPackageForm>(EMPTY_FORM);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [submitState, setSubmitState] = useState<"idle" | "checking" | "ok" | "rejected">("idle");

  const update = (key: keyof EnvironmentPackageForm) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    // preserve user's input on validation failure (SPEC console requirement)
    setForm((f) => ({ ...f, [key]: e.target.value }));
    setFieldErrors((prev) => {
      if (!prev[key]) return prev;
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitState("checking");
    const result = await transport.validateEnvironmentPackage(form);
    if (result.ok) {
      setSubmitState("ok");
      setFieldErrors({});
    } else {
      setSubmitState("rejected");
      const errors: Record<string, string> = {};
      for (const key of result.missingFields) {
        const label = REQUIRED_PACKAGE_FIELDS.find((f) => f.key === key)?.label ?? key;
        errors[key] = `${label} is required or invalid`;
      }
      setFieldErrors(errors); // form values are NOT reset
    }
  };

  if (loading) return <LoadingState label="Loading environment registry…" />;

  return (
    <section aria-labelledby="registry-heading" className="grid gap-6 lg:grid-cols-2">
      <div>
        <h2 id="registry-heading" className="text-base font-semibold text-slate-100">
          Registered environments
        </h2>
        {environments.length === 0 ? (
          <div className="mt-3">
            <EmptyState title="No environments registered">
              An environment package must declare: tool schemas, a policy reference, a trusted evaluator reference,
              and a reset fixture reference. Validation rejects packages missing any of these before a run starts.
            </EmptyState>
          </div>
        ) : (
          <ul className="mt-3 space-y-3">
            {environments.map((env) => (
              <li key={env.environmentId} className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-medium text-slate-100">
                    {env.environmentId} <span className="text-slate-500">v{env.version}</span>
                  </p>
                  <StatusBadge status={env.validationState === "valid" ? "validated" : "invalid"} />
                </div>
                <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 text-[13px] text-slate-400 sm:grid-cols-2">
                  <div className="flex justify-between gap-2">
                    <dt>Tools</dt>
                    <dd className="text-slate-300">{env.toolCount}</dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt>Policy scope</dt>
                    <dd className="text-slate-300">{env.policyScope}</dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt>Evaluator</dt>
                    <dd>
                      <StatusBadge status={env.evaluatorReady ? "live" : "stale"} />
                    </dd>
                  </div>
                </dl>
                {!env.evaluatorReady && (
                  <div className="mt-2">
                    <Banner tone="warn" title="Evaluator unavailable">
                      Learning activation is blocked. Retry after the evaluator recovers.
                    </Banner>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <h2 className="text-base font-semibold text-slate-100">Register an environment package</h2>
        <form onSubmit={submit} noValidate className="mt-3 space-y-3 rounded-xl border border-slate-700 bg-slate-900/60 p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            {REQUIRED_PACKAGE_FIELDS.filter((f) => f.key !== "toolSchemas").map(({ key, label }) => (
              <div key={key}>
                <label htmlFor={`pkg-${key}`} className="block text-xs font-medium text-slate-400">
                  {label}
                </label>
                <input
                  id={`pkg-${key}`}
                  value={form[key]}
                  onChange={update(key)}
                  aria-invalid={fieldErrors[key] ? true : undefined}
                  aria-describedby={fieldErrors[key] ? `pkg-${key}-error` : undefined}
                  className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100 placeholder-slate-500"
                  placeholder={label}
                />
                {fieldErrors[key] && (
                  <p id={`pkg-${key}-error`} role="alert" className="mt-1 text-xs text-rose-400">
                    {fieldErrors[key]}
                  </p>
                )}
              </div>
            ))}
          </div>
          <div>
            <label htmlFor="pkg-toolSchemas" className="block text-xs font-medium text-slate-400">
              Tool schemas (JSON array)
            </label>
            <textarea
              id="pkg-toolSchemas"
              rows={4}
              value={form.toolSchemas}
              onChange={update("toolSchemas")}
              aria-invalid={fieldErrors.toolSchemas ? true : undefined}
              aria-describedby={fieldErrors.toolSchemas ? "pkg-toolSchemas-error" : undefined}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 font-mono text-xs text-slate-100"
              placeholder='[{"name":"inventory.read","effect":"read"}]'
            />
            {fieldErrors.toolSchemas && (
              <p id="pkg-toolSchemas-error" role="alert" className="mt-1 text-xs text-rose-400">
                {fieldErrors.toolSchemas}
              </p>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              className="rounded-md bg-sky-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-60"
              disabled={submitState === "checking"}
            >
              {submitState === "checking" ? "Validating…" : "Validate package"}
            </button>
            {submitState === "rejected" && (
              <p role="alert" className="text-xs text-rose-400">
                Validation failed — your input is preserved below for correction.
              </p>
            )}
            {submitState === "ok" && (
              <p role="status" className="text-xs text-emerald-400">
                ✓ Package validated.
              </p>
            )}
          </div>
        </form>
      </div>
    </section>
  );
}
