ThisBuild / scalaVersion := "3.3.6"
ThisBuild / organization := "io.github.kmizu"
ThisBuild / version := "0.1.0-SNAPSHOT"

lazy val commonSettings = Seq(
  scalacOptions ++= Seq("-deprecation", "-feature", "-Wunused:all"),
  libraryDependencies += "org.scalameta" %% "munit" % "1.1.1" % Test
)

lazy val core = (project in file("."))
  .settings(commonSettings)
  .settings(
    name := "no-matrix-transformer",
    Compile / run / fork := true,
    coverageMinimumStmtTotal := 80,
    coverageFailOnMinimum := true
  )

lazy val docs = (project in file("site-src"))
  .enablePlugins(MdocPlugin)
  .dependsOn(core)
  .settings(
    name := "no-matrix-transformer-docs",
    mdocIn := (ThisBuild / baseDirectory).value / "docs" / "site",
    mdocOut := (ThisBuild / baseDirectory).value / "target" / "mdoc",
    mdocVariables := Map("VERSION" -> version.value)
  )
