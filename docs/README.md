# CHIMERA Documentation

Welcome to the CHIMERA benchmark documentation.

## What is CHIMERA?

**CHIMERA** (Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment) is a comprehensive benchmark for evaluating the meta-cognitive calibration of Large Language Models.

## Quick Links

- [Getting Started](getting-started/installation.md)
- [Conceptual Overview](concepts/calibration.md)
- [API Reference](api/models.md)
- [Configuration Guide](configuration.md)

## Why CHIMERA?

Current AI benchmarks measure *what* models produce, but not *whether models know when they're wrong*. CHIMERA fills this critical gap by evaluating:

1. **Confidence Calibration** - Does stated confidence predict accuracy?
2. **Self-Error Detection** - Can models identify their own mistakes?
3. **Knowledge Boundaries** - Do models know what they don't know?
4. **Self-Correction** - Can models fix errors through introspection?

## Documentation Structure

```
docs/
├── getting-started/
│   ├── installation.md
│   └── quickstart.md
├── concepts/
│   ├── calibration.md
│   ├── introspection.md
│   └── metrics.md
├── api/
│   ├── models.md
│   ├── datasets.md
│   └── evaluation.md
├── tutorials/
│   ├── first-evaluation.md
│   └── custom-tasks.md
└── configuration.md
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

CHIMERA is released under the Apache 2.0 License.
