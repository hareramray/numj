# Releasing numj to Maven Central

Coordinates: `io.github.hareramray:numj` (Java API) and `io.github.hareramray:numj-natives-windows-x86_64`
(native library), parent `io.github.hareramray:numj-parent`. The build is `pom.xml` + `maven/*/pom.xml`. The
release driver is `scripts/release.ps1`, and Maven 3.9.9 is fetched by `scripts/bootstrap.ps1` into `.tools/maven`.

> **A release is permanent.** Maven Central never deletes or replaces a published version. If something is wrong,
> the only fix is a new version. `release.ps1 -Publish` therefore uploads with `autoPublish=false`: Central
> validates the deployment, then it waits in the portal until you press **Publish** (or **Drop** to discard it).

## One-time setup (done by the account owner)

1. **Central Portal account and namespace.** Sign in at <https://central.sonatype.com> with the GitHub account
   `hareramray`. Accounts created through GitHub get the namespace `io.github.hareramray` verified automatically.
   Check *Namespaces* in the portal. If it is not verified, follow the portal's instructions: create a temporary
   public GitHub repository named after the verification key.
2. **User token.** In the portal: *Account → Generate User Token*. Put it in `%USERPROFILE%\.m2\settings.xml`
   (never in the repository):

   ```xml
   <settings>
     <servers>
       <server>
         <id>central</id>
         <username>TOKEN_USERNAME</username>
         <password>TOKEN_PASSWORD</password>
       </server>
     </servers>
   </settings>
   ```
3. **Signing key.** Central requires a PGP signature on every file. Git for Windows ships GnuPG:

   ```powershell
   gpg --full-generate-key                                   # RSA 4096, an expiry date, a passphrase
   gpg --list-secret-keys --keyid-format=long                # note the key id
   gpg --keyserver keyserver.ubuntu.com --send-keys KEYID    # Central verifies signatures against public key servers
   ```
   Keep a backup of the secret key, and a revocation certificate, somewhere safe. During `-Publish`, gpg-agent asks for
   the passphrase. Alternatively, set `MAVEN_GPG_PASSPHRASE` in that shell only, never in a file.

## Each release

1. Merge the release branch into `main`. Check `CHANGELOG.md`, and set the version in all three POMs
   (`pom.xml`, `maven/numj/pom.xml`, `maven/numj-natives-windows-x86_64/pom.xml`).
2. Dry run: `powershell -ExecutionPolicy Bypass -File scripts\release.ps1`. This builds, runs every suite on both
   shipped DLLs, packages, checks the jar contents, and runs the example from the jars alone in an empty directory.
3. Commit, then tag: `git tag v0.2.0 && git push origin main v0.2.0`.
4. Upload: `powershell -ExecutionPolicy Bypass -File scripts\release.ps1 -Publish` (refuses a dirty working tree).
5. Open <https://central.sonatype.com/publishing/deployments>, check the deployment (files, POMs, signatures), and press
   **Publish**. The artifacts appear on `repo1.maven.org` within minutes, and in search within about an hour.
6. Create a GitHub release for the tag, with the changelog section.

## What users add

```xml
<dependency>
  <groupId>io.github.hareramray</groupId>
  <artifactId>numj</artifactId>
  <version>0.2.0</version>
</dependency>
<dependency>
  <groupId>io.github.hareramray</groupId>
  <artifactId>numj-natives-windows-x86_64</artifactId>
  <version>0.2.0</version>
  <scope>runtime</scope>
</dependency>
```

Gradle: `implementation("io.github.hareramray:numj:0.2.0")` and
`runtimeOnly("io.github.hareramray:numj-natives-windows-x86_64:0.2.0")`.
Run with Java 25+ and `--enable-native-access=ALL-UNNAMED`. `java -cp ... numj.NativeInfo` shows which library was
loaded.

## Known limits of 0.2.0 (state them in the release notes)

* Natives exist for **Windows x86-64 only**. On Linux or macOS, numj throws `UnsatisfiedLinkError` with an explanation
  (the code paths exist but are unbuilt and untested).
* Pre-1.0 API: `F64Array.flatten()` changes from a view to a copy in 0.3 (see MIGRATION.md).
* The POM lists the developer without an e-mail address. Adding one publishes it permanently.
