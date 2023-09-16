<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Stochastic Variational Inference with Mixture Models>>

  If use gumbel:

  - gradient high variance

  - sample now differentiable

  - expectation is poorly approximate

  - gradient now biased

  - log prob cannot be computed exactly

  \;

  If use categorical:

  - gradient low variance

  - but computational cost scales with number of mixture components (in log
  prob computation especially, and ELBO computation)

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>